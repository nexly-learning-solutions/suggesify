
#include "../src/cutlass_kernels/cutlass_preprocessors.h"
#include "../common/assert.h"
#include "../common/cudaBf16Wrapper.h"
#include "../common/stringUtils.h"

#include "cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

using namespace suggestify::common;

namespace suggestify
{
namespace kernels
{
namespace cutlass_kernels
{

struct LayoutDetails
{
    enum class Layout
    {
        UNKNOWN,
        ROW_MAJOR,
        COLUMN_MAJOR
    };

    Layout layoutB = Layout::UNKNOWN;
    int rows_per_column_tile = 1;
    int columns_interleaved = 1;

    bool uses_imma_ldsm = false;
};

template <typename Layout>
struct getLayoutDetails
{
};

template <>
struct getLayoutDetails<cutlass::layout::RowMajor>
{
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::ROW_MAJOR;
        return layout_details;
    }
};

template <>
struct getLayoutDetails<cutlass::layout::ColumnMajor>
{
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
        return layout_details;
    }
};

template <int RowsPerTile, int ColumnsInterleaved>
struct getLayoutDetails<cutlass::layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>>
{
    LayoutDetails operator()()
    {
        LayoutDetails layout_details;
        layout_details.layoutB = LayoutDetails::Layout::COLUMN_MAJOR;
        layout_details.rows_per_column_tile = RowsPerTile;
        layout_details.columns_interleaved = ColumnsInterleaved;
        return layout_details;
    }
};

template <typename cutlassArch, typename TypeA, typename TypeB>
LayoutDetails getLayoutDetailsForArchAndQuantType()
{

    using CompileTraits = cutlass::gemm::kernel::LayoutDetailsB<TypeA, TypeB, cutlassArch>;
    using LayoutB = typename CompileTraits::Layout;
    using MmaOperator = typename CompileTraits::Operator;
    LayoutDetails details = getLayoutDetails<LayoutB>()();
    details.uses_imma_ldsm = std::is_same<MmaOperator, cutlass::arch::OpMultiplyAddDequantizeInterleavedBToA>::value;
    return details;
}

template <typename cutlassArch>
LayoutDetails getLayoutDetailsForArch(QuantType quant_type)
{
    int const bits_per_weight_element = get_weight_quant_bits(quant_type);
    LayoutDetails details;
    switch (quant_type)
    {
    case QuantType::W8_A16:
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::half_t, uint8_t>();
        break;
    case QuantType::W4_A16:
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::half_t, cutlass::uint4b_t>();
        break;
    case QuantType::W4_AFP8:
        details = getLayoutDetailsForArchAndQuantType<cutlassArch, cutlass::float_e4m3_t, cutlass::uint4b_t>();
        break;
    default: THROW("Unsupported quantization type");
    }
    return details;
}

LayoutDetails getLayoutDetailsForTransform(QuantType quant_type, int arch)
{
    if (arch >= 75 && arch < 80)
    {
        return getLayoutDetailsForArch<cutlass::arch::Sm75>(quant_type);
    }
    else if (arch >= 80 && arch < 90)
    {
        return getLayoutDetailsForArch<cutlass::arch::Sm80>(quant_type);
    }
    else if (arch == 90)
    {
        return getLayoutDetailsForArch<cutlass::arch::Sm90>(quant_type);
    }
    else
    {
        CHECK_WITH_INFO(false, "Unsupported Arch");
        return LayoutDetails();
    }
}

std::vector<int> get_permutation_map(QuantType quant_type)
{

    if (quant_type == QuantType::W8_A16)
    {
        return {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    }
    else if (quant_type == QuantType::W4_A16)
    {
        return {0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15,
            22, 23, 30, 31};
    }
    else if (quant_type == QuantType::W4_AFP8)
    {
        return {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23, 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15,
            28, 29, 30, 31};
    }
    else
    {
        THROW("Invalid quantization type for LDSM permutation");
    }
}

void permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type, int64_t const arch_version)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    std::vector<int> row_permutation = get_permutation_map(quant_type);

    CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    int const BITS_PER_ELT = get_weight_quant_bits(quant_type);
    int const K = 16 / BITS_PER_ELT;
    int const ELTS_PER_BYTE = 8 / BITS_PER_ELT;
    int const ELTS_PER_REG = 32 / BITS_PER_ELT;

    uint32_t const* input_byte_ptr = reinterpret_cast<uint32_t const*>(quantized_tensor);
    uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

    int MMA_SHAPE_N = 8;
    int B_ROWS_PER_MMA = 8 * K;
    int const elts_in_int32 = 32 / BITS_PER_ELT;

    int const num_vec_cols = num_cols / elts_in_int32;

    CHECK_WITH_INFO(
        arch_version >= 75, "Unsupported Arch. Pre-volta not supported. Column interleave not needed on Volta.");

    CHECK_WITH_INFO(num_rows % B_ROWS_PER_MMA == 0,
        fmtstr("Invalid shape for quantized tensor. Number of rows of quantized matrix must be a multiple of %d",
            B_ROWS_PER_MMA));
    CHECK_WITH_INFO(num_cols % MMA_SHAPE_N == 0,
        fmtstr("Invalid shape for quantized tensor. On turing/Ampere, the number of cols must be a multiple of %d.",
            MMA_SHAPE_N));

    CHECK_WITH_INFO(size_t(B_ROWS_PER_MMA) == row_permutation.size(), "Unexpected number of LDSM rows permuted.");

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int64_t matrix_offset = expert * int64_t(num_rows) * int64_t(num_vec_cols);
        for (int base_row = 0; base_row < num_rows; base_row += B_ROWS_PER_MMA)
        {
            for (int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row)
            {

                for (int write_col = 0; write_col < num_vec_cols; ++write_col)
                {
                    int const write_row = base_row + tile_row;
                    int const tile_read_row = row_permutation[tile_row];
                    int const read_row = base_row + tile_read_row;
                    int const read_col = write_col;

                    const int64_t read_offset = matrix_offset + int64_t(read_row) * num_vec_cols + read_col;
                    const int64_t write_offset = matrix_offset + int64_t(write_row) * num_vec_cols + write_col;

                    output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
                }
            }
        }
    }
}

template <QuantType quant_type>
void subbyte_transpose_impl(
    int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor, std::vector<size_t> const& shape)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    constexpr int bits_per_elt = get_weight_quant_bits(quant_type);

    CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    const size_t col_bytes = num_cols * bits_per_elt / 8;
    const size_t col_bytes_trans = num_rows * bits_per_elt / 8;
    const size_t num_bytes = size_t(num_experts) * num_rows * col_bytes;

    uint8_t const* input_byte_ptr = reinterpret_cast<uint8_t const*>(quantized_tensor);
    uint8_t* output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

    static constexpr int ELTS_PER_BYTE = 8 / bits_per_elt;

    static constexpr int M_TILE_L1 = 64;
    static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;
    uint8_t cache_buf[M_TILE_L1][N_TILE_L1];

    static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

    CHECK_WITH_INFO(!(col_bytes_trans % VECTOR_WIDTH) && !(col_bytes % VECTOR_WIDTH),
        fmtstr("Number of bytes for rows and cols must be a multiple of %d. However, num_rows_bytes = %ld and "
               "num_col_bytes = %ld.",
            VECTOR_WIDTH, col_bytes_trans, col_bytes));

    int const num_m_tiles = (num_rows + M_TILE_L1 - 1) / M_TILE_L1;
    int const num_n_tiles = (col_bytes + N_TILE_L1 - 1) / N_TILE_L1;

    for (size_t expert = 0; expert < num_experts; ++expert)
    {
        const size_t matrix_offset = expert * num_rows * col_bytes;
        for (size_t row_tile_start = 0; row_tile_start < num_rows; row_tile_start += M_TILE_L1)
        {
            for (size_t col_tile_start_byte = 0; col_tile_start_byte < col_bytes; col_tile_start_byte += N_TILE_L1)
            {

                int const row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
                int const col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);

                for (int ii = 0; ii < M_TILE_L1; ++ii)
                {
                    int const row = row_tile_start + ii;

                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                    {
                        int const col = col_tile_start_byte + jj;

                        const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

                        if (row < row_limit && col < col_limit)
                        {
                            for (int v = 0; v < VECTOR_WIDTH; ++v)
                            {
                                cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
                            }
                        }
                    }
                }

                if constexpr (bits_per_elt == 8)
                {
                    for (int ii = 0; ii < M_TILE_L1; ++ii)
                    {
                        for (int jj = ii + 1; jj < N_TILE_L1; ++jj)
                        {
                            std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
                        }
                    }
                }
                else if constexpr (bits_per_elt == 4)
                {

                    for (int ii = 0; ii < M_TILE_L1; ++ii)
                    {
                        for (int jj = ii + 1; jj < M_TILE_L1; ++jj)
                        {
                            int const ii_byte = ii / ELTS_PER_BYTE;
                            int const ii_bit_offset = ii % ELTS_PER_BYTE;

                            int const jj_byte = jj / ELTS_PER_BYTE;
                            int const jj_bit_offset = jj % ELTS_PER_BYTE;

                            uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
                            uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

                            cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
                            cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

                            cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
                            cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
                        }
                    }
                }
                else
                {
                    CHECK_WITH_INFO(false, "Unsupported quantization type.");
                }

                const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
                const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

                int const row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);
                int const col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

                for (int ii = 0; ii < M_TILE_L1; ++ii)
                {
                    int const row = row_tile_start_trans + ii;
                    for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                    {
                        int const col = col_tile_start_byte_trans + jj;

                        const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

                        if (row < row_limit_trans && col < col_limit_trans)
                        {
                            for (int v = 0; v < VECTOR_WIDTH; ++v)
                            {
                                output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
                            }
                        }
                    }
                }
            }
        }
    }
}

void subbyte_transpose(int8_t* transposed_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (quant_type == QuantType::W8_A16)
    {
        subbyte_transpose_impl<QuantType::W8_A16>(transposed_quantized_tensor, quantized_tensor, shape);
    }
    else if (quant_type == QuantType::W4_A16)
    {
        subbyte_transpose_impl<QuantType::W4_A16>(transposed_quantized_tensor, quantized_tensor, shape);
    }
    else if (quant_type == QuantType::W4_AFP8)
    {
        subbyte_transpose_impl<QuantType::W4_AFP8>(transposed_quantized_tensor, quantized_tensor, shape);
    }
    else
    {
        CHECK_WITH_INFO(false, "Invalid quant_type");
    }
}

void add_bias_and_interleave_int8s_inplace(int8_t* int8_tensor, const size_t num_elts)
{
    for (int ii = 0; ii < num_elts; ++ii)
    {
        int8_tensor[ii] = int8_t(int(int8_tensor[ii]) + 128);
    }


    CHECK_WITH_INFO(num_elts % 4 == 0, "Dimensions of int8 tensor must be a multiple of 4 for register relayout");
    for (size_t base = 0; base < num_elts; base += 4)
    {
        std::swap(int8_tensor[base + 1], int8_tensor[base + 2]);
    }
}

void add_bias_and_interleave_int4s_inplace(int8_t* packed_int4_tensor, const size_t num_elts)
{
    int const num_bytes = num_elts / 2;

    for (size_t ii = 0; ii < num_bytes; ++ii)
    {
        int8_t transformed_packed_int4s = 0;
        int8_t transformed_first_elt
            = (int8_t(packed_int4_tensor[ii] << 4) >> 4) + 8;
        int8_t transformed_second_elt = (packed_int4_tensor[ii] >> 4) + 8;

        CHECK_WITH_INFO(
            transformed_first_elt >= 0 && transformed_first_elt <= 15, "Illegal result for int4 transform (first elt)");
        CHECK_WITH_INFO(transformed_second_elt >= 0 && transformed_second_elt <= 15,
            "Illegal result for int4 transform (second elt)");

        transformed_packed_int4s |= transformed_first_elt;
        transformed_packed_int4s |= (transformed_second_elt << 4);
        packed_int4_tensor[ii] = transformed_packed_int4s;
    }


    CHECK_WITH_INFO(num_bytes % 4 == 0, "Dimensions of int4 tensor must be a multiple of 8 for register relayout");
    const size_t num_registers = num_bytes / 4;

    uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int4_tensor);
    for (size_t ii = 0; ii < num_registers; ++ii)
    {
        const uint32_t current_register = register_ptr[ii];
        uint32_t transformed_register = 0;

        for (int dest_idx = 0; dest_idx < 8; ++dest_idx)
        {
            int const src_idx = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
            int const src_shift = 4 * src_idx;
            int const dest_shift = 4 * dest_idx;

            const uint32_t src_bits = (current_register >> src_shift) & 0xF;
            transformed_register |= (src_bits << dest_shift);
        }
        register_ptr[ii] = transformed_register;
    }
}

void add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (quant_type == QuantType::W8_A16)
    {
        add_bias_and_interleave_int8s_inplace(tensor, num_elts);
    }
    else if (quant_type == QuantType::W4_A16 || quant_type == QuantType::W4_AFP8)
    {
        add_bias_and_interleave_int4s_inplace(tensor, num_elts);
    }
    else
    {
        CHECK_WITH_INFO(false, "Invalid quantization type for interleaving.");
    }
}

void interleave_column_major_tensor(int8_t* interleaved_quantized_tensor, int8_t const* quantized_tensor,
    std::vector<size_t> const& shape, QuantType quant_type, LayoutDetails details)
{
    LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    int const BITS_PER_ELT = get_weight_quant_bits(quant_type);
    int const elts_in_int32 = 32 / BITS_PER_ELT;

    int const rows_per_tile = details.rows_per_column_tile;

    CHECK_WITH_INFO(!(num_rows % elts_in_int32),
        fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", elts_in_int32, num_rows));

    uint32_t const* input_byte_ptr = reinterpret_cast<uint32_t const*>(quantized_tensor);
    uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

    CHECK_WITH_INFO(!(num_rows % rows_per_tile),
        fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", rows_per_tile, num_rows));

    int const num_vec_rows = num_rows / elts_in_int32;
    int const vec_rows_per_tile = rows_per_tile / elts_in_int32;
    int const interleave = details.columns_interleaved;

    for (int expert = 0; expert < num_experts; ++expert)
    {
        const int64_t matrix_offset = expert * int64_t(num_vec_rows) * int64_t(num_cols);
        for (int read_col = 0; read_col < num_cols; ++read_col)
        {
            const int64_t write_col = read_col / interleave;
            for (int base_vec_row = 0; base_vec_row < num_vec_rows; base_vec_row += vec_rows_per_tile)
            {
                for (int vec_read_row = base_vec_row;
                     vec_read_row < std::min(num_vec_rows, base_vec_row + vec_rows_per_tile); ++vec_read_row)
                {
                    const int64_t vec_write_row = interleave * base_vec_row
                        + vec_rows_per_tile * (read_col % interleave) + vec_read_row % vec_rows_per_tile;

                    const int64_t read_offset = matrix_offset + int64_t(read_col) * num_vec_rows + vec_read_row;
                    const int64_t write_offset
                        = matrix_offset + int64_t(write_col) * num_vec_rows * interleave + vec_write_row;
                    output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
                }
            }
        }
    }
}

void preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t const* row_major_quantized_weight,
    std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave)
{
    int arch = getSMVersion();
    if (force_interleave && arch == 90)
    {
        arch = 80;
    }
    LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

    CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");

    size_t num_elts = 1;
    for (auto const& dim : shape)
    {
        num_elts *= dim;
    }

    const size_t num_bytes = num_elts * get_weight_quant_bits(quant_type) / 8;

    std::vector<int8_t> src_buf(num_bytes);
    std::vector<int8_t> dst_buf(num_bytes);
    std::copy(row_major_quantized_weight, row_major_quantized_weight + num_bytes, src_buf.begin());

    if (details.uses_imma_ldsm)
    {
        permute_B_rows_for_mixed_gemm(dst_buf.data(), src_buf.data(), shape, quant_type, arch);
        src_buf.swap(dst_buf);
    }

    if (details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR)
    {
        subbyte_transpose(dst_buf.data(), src_buf.data(), shape, quant_type);
        src_buf.swap(dst_buf);
    }

    if (details.columns_interleaved > 1)
    {
        interleave_column_major_tensor(dst_buf.data(), src_buf.data(), shape, quant_type, details);
        src_buf.swap(dst_buf);
    }

    if (arch >= 70 && arch < 90)
    {
        add_bias_and_interleave_quantized_tensor_inplace(src_buf.data(), num_elts, quant_type);
    }
    std::copy(src_buf.begin(), src_buf.end(), preprocessed_quantized_weight);
}


template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, int8_t* unprocessed_quantized_weight,
    ComputeType* scale_ptr, WeightType const* input_weight_ptr, std::vector<size_t> const& shape, QuantType quant_type,
    bool force_interleave)
{

    CHECK_WITH_INFO(processed_quantized_weight, "Processed quantized tensor is NULL");
    CHECK_WITH_INFO(scale_ptr, "Scale output pointer is NULL");
    CHECK_WITH_INFO(input_weight_ptr, "Input weight pointer is NULL");

    CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    int const bits_in_type = get_weight_quant_bits(quant_type);
    int const bytes_per_out_col = num_cols * bits_in_type / 8;

    int const bits_per_weigtht_element = get_weight_quant_bits(quant_type);

    std::vector<int8_t> weight_buf;
    if (unprocessed_quantized_weight == nullptr)
    {
        weight_buf.resize(num_experts * num_rows * num_cols);
        unprocessed_quantized_weight = weight_buf.data();
    }

    int const input_mat_size = num_rows * num_cols;
    int const quantized_mat_size = num_rows * bytes_per_out_col;
    float const quant_range_scale = 1.f / float(1 << (bits_in_type - 1));

    std::vector<float> per_col_max(num_cols);

    for (int expert = 0; expert < num_experts; ++expert)
    {
        WeightType const* current_weight = input_weight_ptr + expert * input_mat_size;
        int8_t* current_quantized_weight = unprocessed_quantized_weight + expert * quantized_mat_size;

        for (int jj = 0; jj < num_cols; ++jj)
        {
            per_col_max[jj] = 0.f;
        }

        for (int ii = 0; ii < num_rows; ++ii)
        {
            WeightType const* current_weight_row = current_weight + ii * num_cols;
            for (int jj = 0; jj < num_cols; ++jj)
            {
                per_col_max[jj] = std::max(per_col_max[jj], std::abs(float(current_weight_row[jj])));
            }
        }

        ComputeType* current_scales = scale_ptr + expert * num_cols;
        for (int jj = 0; jj < num_cols; ++jj)
        {
            per_col_max[jj] *= quant_range_scale;
            current_scales[jj] = ComputeType(per_col_max[jj]);
        }

        for (int ii = 0; ii < num_rows; ++ii)
        {
            int8_t* current_quantized_weight_row = current_quantized_weight + ii * bytes_per_out_col;
            WeightType const* current_weight_row = current_weight + ii * num_cols;
            for (int jj = 0; jj < bytes_per_out_col; ++jj)
            {

                if (bits_per_weigtht_element == 8)
                {
                    float const col_scale = per_col_max[jj];
                    float const weight_elt = float(current_weight_row[jj]);
                    float const scaled_weight = (col_scale != 0.0f) ? round(weight_elt / col_scale) : 0.0f;
                    const int8_t clipped_weight = int8_t(std::max(-128.f, std::min(127.f, scaled_weight)));
                    current_quantized_weight_row[jj] = clipped_weight;
                }
                else if (bits_per_weigtht_element == 4)
                {

                    int8_t packed_int4s = 0;
                    for (int packed_idx = 0; packed_idx < 2; ++packed_idx)
                    {
                        int const input_idx = 2 * jj + packed_idx;
                        if (input_idx < num_cols)
                        {
                            float const col_scale = per_col_max[input_idx];
                            float const weight_elt = float(current_weight_row[input_idx]);
                            float const scaled_weight = (col_scale != 0.0f) ? round(weight_elt / col_scale) : 0.0f;
                            int int_weight = int(scaled_weight);
                            const int8_t clipped_weight = std::max(-8, std::min(7, int_weight));

                            packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
                        }
                    }
                    current_quantized_weight_row[jj] = packed_int4s;
                }
                else
                {
                    CHECK_WITH_INFO(false, "Unsupported quantization type");
                }
            }
        }
    }

    preprocess_weights_for_mixed_gemm(
        processed_quantized_weight, unprocessed_quantized_weight, shape, quant_type, force_interleave);
}

template void symmetric_quantize<half, float>(
    int8_t*, int8_t*, half*, float const*, std::vector<size_t> const&, QuantType, bool);

template void symmetric_quantize<half, half>(
    int8_t*, int8_t*, half*, half const*, std::vector<size_t> const&, QuantType, bool);

#ifdef ENABLE_BF16
template void symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(
    int8_t*, int8_t*, __nv_bfloat16*, __nv_bfloat16 const*, std::vector<size_t> const&, QuantType, bool);

template void symmetric_quantize<__nv_bfloat16, float>(
    int8_t*, int8_t*, __nv_bfloat16*, float const*, std::vector<size_t> const&, QuantType, bool);
#endif

template <typename ComputeType, typename WeightType>
void symmetric_quantize(int8_t* processed_quantized_weight, ComputeType* scale_ptr, WeightType const* input_weight_ptr,
    std::vector<size_t> const& shape, QuantType quant_type, bool force_interleave)
{
    symmetric_quantize(
        processed_quantized_weight, nullptr, scale_ptr, input_weight_ptr, shape, quant_type, force_interleave);
}

template void symmetric_quantize<float, float>(
    int8_t*, float*, float const*, std::vector<size_t> const&, QuantType, bool);

template void symmetric_quantize<half, float>(
    int8_t*, half*, float const*, std::vector<size_t> const&, QuantType, bool);

template void symmetric_quantize<half, half>(int8_t*, half*, half const*, std::vector<size_t> const&, QuantType, bool);

#ifdef ENABLE_BF16
template void symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(
    int8_t*, __nv_bfloat16*, __nv_bfloat16 const*, std::vector<size_t> const&, QuantType, bool);

template void symmetric_quantize<__nv_bfloat16, half>(
    int8_t*, __nv_bfloat16*, half const*, std::vector<size_t> const&, QuantType, bool);

template void symmetric_quantize<half, __nv_bfloat16>(
    int8_t*, half*, __nv_bfloat16 const*, std::vector<size_t> const&, QuantType, bool);

template void symmetric_quantize<__nv_bfloat16, float>(
    int8_t*, __nv_bfloat16*, float const*, std::vector<size_t> const&, QuantType, bool);
#endif

}
}
}
