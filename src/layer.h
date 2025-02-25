#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <span>
#include <ncFile.h>
#include "Types.h"
#include <functional>

using ActivationFunction = std::function<void(void*, uint64_t)>;

class LayerDescriptor;

class Layer {
public:
    friend class Network;
    friend class Weight;
    friend Network* LoadNeuralNetworkNetCDF(const std::string& fname, int batch);
    enum Kind
    {
        Input,
        Hidden,
        Output,
        Target,
    };

    static std::pair<Layer::Kind, std::string> _sKindPair[];
    static const std::map<Kind, std::string> _sKindMap;

    enum Type
    {
        FullyConnected,
        Convolutional,
        Pooling,
        LSTM
    };

    static std::pair<Layer::Type, std::string> _sTypePair[];
    static const std::map<Type, std::string> _sTypeMap;

    enum Attributes
    {
        None = 0x0,
        Sparse = 0x1,
        Denoising = 0x2,
        BatchNormal = 0x4,
    };

    static std::pair<Layer::Attributes, std::string> _sAttributesPair[];
    static const std::map<Attributes, std::string> _sAttributesMap;

    enum Parallelization {
        Data,
        Model,
        Serial,
    };

    static std::pair<Layer::Parallelization, std::string> _sParallelizationPair[];
    static const std::map<Parallelization, std::string> _sParallelizationMap;


private:
    const std::string _name;
    const Kind _kind;
    const Type _type;
    const uint32_t _attributes;
    PoolingFunction _poolingFunction;
    std::string _dataSet;
    DataSetBase* _pDataSet;
    std::vector<std::string> _vSource;
    std::vector<std::string> _vSkip;
    uint32_t _Nx;
    uint32_t _Ny;
    uint32_t _Nz;
    uint32_t _Nw;
    uint32_t _stride;
    uint32_t _localStride;
    uint32_t _maxLocalStride;
    uint32_t _strideBN;
    uint32_t _batch;
    uint32_t _localBatch;
    uint32_t _deltaUpdateCount;
    uint32_t _unitUpdateCount;
    uint32_t _dimensions;
    uint32_t _minX;
    uint32_t _maxX;
    WeightInitialization _weightInit;
    float _weightInitScale;
    float _biasInit;
    float _RELUSlope;
    float _ELUAlpha;
    float _SELULambda;
    bool _bBatchNormalization;
    const uint32_t _kernelX;
    const uint32_t _kernelY;
    const uint32_t _kernelZ;
    const uint32_t _kernelStrideX;
    const uint32_t _kernelStrideY;
    const uint32_t _kernelStrideZ;
    const uint32_t _kernelPaddingX;
    const uint32_t _kernelPaddingY;
    const uint32_t _kernelPaddingZ;
    const uint32_t _kernelDimensions;
    const Activation _activation;
    const float _pDropout;
    bool _bSparse;
    bool _bFastSparse;
    float _sparsenessPenalty_p;
    float _sparsenessPenalty_beta;
    const bool _bDenoising;
    float _weightNorm;
    float _deltaNorm;
    Parallelization _parallelization;
    bool _bTransposeParallelization;
    bool _bDirty;
    cudnnTensorDescriptor_t _scaleBiasMeanVarDescBN;
    cudnnTensorDescriptor_t _tensorDescriptorBN;
    cudnnTensorDescriptor_t _tensorDescriptor;
    cudnnTensorDescriptor_t _oddBatchTensorDescriptor;
    uint32_t _oddBatch;
    cudnnPoolingDescriptor_t _poolingDescriptor;
    cudnnLRNDescriptor_t _LRNDescriptor;
    std::vector<Layer*> _vIncomingLayer;
    std::vector<Weight*> _vIncomingWeight;
    std::vector<Layer*> _vOutgoingLayer;
    std::vector<Weight*> _vOutgoingWeight;
    std::vector<Layer*> _vIncomingLargerLayer;
    std::vector<Weight*> _vIncomingLargerWeight;
    std::vector<Layer*> _vOutgoingLargerLayer;
    std::vector<Weight*> _vOutgoingLargerWeight;
    std::vector<Layer*> _vIncomingSkip;
    std::vector<Layer*> _vOutgoingSkip;
    std::vector<float> _vUnit;
    std::vector<float> _vDelta;
    std::vector<float> _vBuffer1;
    std::vector<float> _vBuffer2;
    std::unique_ptr<GpuBuffer<float>> _pbUnit;
    std::unique_ptr<GpuBuffer<float>> _pbDelta;
    std::unique_ptr<GpuBuffer<float>> _pbDropout;
    std::unique_ptr<GpuBuffer<float>> _pbBuffer1;
    std::unique_ptr<GpuBuffer<float>> _pbBuffer2;
    std::unique_ptr<GpuBuffer<float>> _pbDeltaBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleGradientBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradientBN;
    std::unique_ptr<GpuBuffer<float>> _pbUnitBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbScaleGradientVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbBiasGradientVelocityBN;
    std::unique_ptr<GpuBuffer<float>> _pbRunningMeanBN;
    std::unique_ptr<GpuBuffer<float>> _pbRunningVarianceBN;
    std::unique_ptr<GpuBuffer<float>> _pbSaveMeanBN;
    std::unique_ptr<GpuBuffer<float>> _pbSaveInvVarianceBN;
    uint32_t _bnCalls;
    int32_t _priority;
    Layer(LayerDescriptor& l, uint32_t batch);
    ~Layer();
    /// <summary>
    /// Destroys all CuDNN descriptors associated with the layer.
    /// </summary>
    void DestroyCudnnDescriptors();

    /// <summary>
    /// Destroys a specific CuDNN tensor descriptor.
    /// </summary>
    /// <param name="descriptor">Reference to the CuDNN tensor descriptor.</param>
    void DestroyCudnnDescriptor(cudnnTensorDescriptor_t& descriptor);

    /// <summary>
    /// Resets the buffers used for batch normalization.
    /// </summary>
    void ResetBatchNormalizationBuffers();

    /// <summary>
    /// Destroys the CuDNN pooling descriptor.
    /// </summary>
    void DestroyPoolingDescriptor();

    /// <summary>
    /// Allocates memory for the layer.
    /// </summary>
    /// <param name="validate">Flag indicating whether to validate the allocation.</param>
    void Allocate(bool validate);

    /// <summary>
    /// Deallocates memory for the layer.
    /// </summary>
    void Deallocate();

    /// <summary>
    /// Sets the batch size for the layer.
    /// </summary>
    /// <param name="batch">Batch size to set.</param>
    void SetBatch(uint32_t batch);

    /// <summary>
    /// Refreshes the parallelization settings.
    /// </summary>
    void RefreshParallelization();

    /// <summary>
    /// Refreshes the state of the network for a given training mode.
    /// </summary>
    /// <param name="pNetwork">Pointer to the network.</param>
    /// <param name="trainingMode">Training mode (e.g., training or inference).</param>
    /// <param name="validate">Flag indicating whether to validate the state refresh.</param>
    void RefreshState(Network* pNetwork, TrainingMode trainingMode, bool validate);

    /// <summary>
    /// Loads a batch for prediction at a specified position.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to load.</param>
    void LoadPredictionBatch(uint32_t position, uint32_t batch);

    /// <summary>
    /// Loads a batch for training at a specified position.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to load.</param>
    void LoadTrainingBatch(uint32_t position, uint32_t batch);

    /// <summary>
    /// Loads a batch for validation at a specified position.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to load.</param>
    void LoadValidationBatch(uint32_t position, uint32_t batch);

    /// <summary>
    /// Performs forward propagation for the layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="bTraining">Flag indicating whether training is active.</param>
    void ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining = false);

    /// <summary>
    /// Performs forward propagation for a fully connected layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="bTraining">Flag indicating whether training is active.</param>
    void ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining);

    /// <summary>
    /// Performs forward propagation for a convolutional layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="bTraining">Flag indicating whether training is active.</param>
    void ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining);

    /// <summary>
    /// Performs forward propagation for a pooling layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="bTraining">Flag indicating whether training is active.</param>
    void ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining);

    /// <summary>
    /// Calculates the activation for the current batch.
    /// </summary>
    /// <param name="batch">Batch size to process.</param>
    void CalculateActivation(uint32_t batch);

    /// <summary>
    /// Calculates the error for a specific position and batch using the given error function.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="ef">Error function to use.</param>
    /// <returns>Error calculated for the specified position and batch.</returns>
    float CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef);

    /// <summary>
    /// Calculates the delta for the output layer during backpropagation.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="ef">Error function used during forward propagation.</param>
    void CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef);

    /// <summary>
    /// Performs backpropagation for the layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    void BackPropagate(uint32_t position, uint32_t batch);

    /// <summary>
    /// Performs backpropagation for a convolutional layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    void BackPropagateConvolutional(uint32_t position, uint32_t batch);

    /// <summary>
    /// Performs backpropagation for a pooling layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    void BackPropagatePooling(uint32_t position, uint32_t batch);

    /// <summary>
    /// Calculates dropout for the current batch.
    /// </summary>
    /// <param name="batch">Batch size to process.</param>
    void CalculateDropout(uint32_t batch);

    /// <summary>
    /// Performs backpropagation for a fully connected layer.
    /// </summary>
    /// <param name="position">Position in the dataset.</param>
    /// <param name="batch">Batch size to process.</param>
    void BackPropagateFullyConnected(uint32_t position, uint32_t batch);

    /// <summary>
    /// Updates the weights of the layer based on the training mode and other parameters.
    /// </summary>
    /// <param name="trainingMode">Training mode (e.g., SGD).</param>
    /// <param name="batch">Batch size used during training.</param>
    /// <param name="alpha">Learning rate.</param>
    /// <param name="lambda">Weight decay (L2 regularization).</param>
    /// <param name="lambda1">Weight decay for bias (L1 regularization).</param>
    /// <param name="mu">Momentum for weight updates.</param>
    /// <param name="mu1">Momentum for bias updates.</param>
    /// <param name="t">Time step or epoch.</param>
    void UpdateWeights(TrainingMode trainingMode, uint32_t batch, float alpha, float lambda, float lambda1, float mu, float mu1, float t);

    /// <summary>
    /// Generates denoising data for the layer.
    /// </summary>
    void GenerateDenoisingData();

    /// <summary>
    /// Reduces data across multiple nodes in a distributed setup.
    /// </summary>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="stride">Stride between elements in the buffer.</param>
    /// <param name="pBuffer">Pointer to the data buffer.</param>
    /// <param name="localStride">Stride within the local buffer.</param>
    /// <param name="updateCount">Number of updates to perform.</param>
    void Reduce(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride, uint32_t updateCount);

    /// <summary>
    /// Gathers data across multiple nodes in a distributed setup.
    /// </summary>
    /// <param name="batch">Batch size to process.</param>
    /// <param name="stride">Stride between elements in the buffer.</param>
    /// <param name="pBuffer">Pointer to the data buffer.</param>
    /// <param name="localStride">Stride within the local buffer.</param>
    void Gather(uint32_t batch, uint32_t stride, float* pBuffer, uint32_t localStride);

    /// <summary>
    /// Clears the weight updates for the layer.
    /// </summary>
    void ClearUpdates();

    /// <summary>
    /// Dumps data to a specified file.
    /// </summary>
    /// <param name="fname">File name to dump data into.</param>
    /// <param name="pData">Pointer to the data to dump.</param>
    void Dump(std::string fname, float* pData);

    /// <summary>
    /// Writes the network configuration to a NetCDF file.
    /// </summary>
    /// <param name="nc">NetCDF file object.</param>
    /// <param name="index">Index of the layer within the network.</param>
    /// <returns>True if writing to NetCDF was successful, false otherwise.</returns>
    bool WriteNetCDF(netCDF::NcFile& nc, uint32_t index);
    void NamedEntityRecognition(uint32_t batch, uint32_t sequenceLength, uint32_t numEntities);
    float* GetIncomingUnitBuffer()
    {
        if (_bBatchNormalization)
            return _pbUnitBN ? _pbUnitBN->_pDevData : NULL;
        else
            return _pbUnit ? _pbUnit->_pDevData : NULL;
    }
    float* GetUnitBuffer() { return _pbUnit ? _pbUnit->_pDevData : NULL; }
    float* GetIncomingDeltaBuffer()
    {
        if (_bBatchNormalization)
            return _pbDeltaBN ? _pbDeltaBN->_pDevData : NULL;
        else
            return _pbDelta ? _pbDelta->_pDevData : NULL;
    }
    float* GetDeltaBuffer() { return _pbDelta ? _pbDelta->_pDevData : NULL; }
    uint64_t GetBufferSize() { return _batch * _stride; }
    cudnnTensorDescriptor_t getTensorDescriptor(uint32_t batch);
    cudnnTensorDescriptor_t getTensorDescriptorBN(uint32_t batch);

public:
    const std::string& GetName() const;

    const std::string& GetDataSetName() const;

    Layer::Kind GetKind() const;
    Layer::Type GetType() const;
    uint32_t GetAttributes() const;

    DataSetBase* GetDataSet() const;

    uint32_t GetNumDimensions() const;

    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetDimensions() const;
    virtual std::vector<double> forward(const std::vector<double>& input) const = 0;
    virtual std::vector<double> backward(const std::vector<double>& error) const = 0;
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLocalDimensions() const;
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelDimensions() const;
    std::tuple<uint32_t, uint32_t, uint32_t> GetKernelStride() const;
    bool GetUnits(std::vector<float>& vUnit);
    bool GetUnits(float* pUnit);
    bool GetUnits(std::span<float> units);
    bool SetUnits(const std::vector<float>& vUnit);
    bool GetDeltas(std::vector<float>& vUnit);
    bool GetDeltas(float* pUnit);
    bool SetDeltas(const std::vector<float>& vUnit);

};


std::ostream& operator<< (std::ostream& out, Layer::Kind& k);
std::ostream& operator<< (std::ostream& out, Layer::Type& t);
std::ostream& operator<< (std::ostream& out, Layer::Parallelization& p);
std::ostream& operator<< (std::ostream& out, Layer::Attributes& a);

struct LayerDescriptor
{
    /// <summary>
    /// Name of the layer.
    /// </summary>
    std::string _name;

    /// <summary>
    /// Kind of layer (enum Layer::Kind).
    /// </summary>
    Layer::Kind _kind;

    /// <summary>
    /// Type of layer (enum Layer::Type).
    /// </summary>
    Layer::Type _type;

    /// <summary>
    /// Pooling function used by the layer.
    /// </summary>
    PoolingFunction _poolingFunction;

    /// <summary>
    /// Dataset associated with the layer.
    /// </summary>
    std::string _dataSet;

    /// <summary>
    /// Sources for the layer.
    /// </summary>
    std::vector<std::string> _vSource;

    /// <summary>
    /// Skip connections for the layer.
    /// </summary>
    std::vector<std::string> _vSkip;

    /// <summary>
    /// Size of the layer in the x-dimension.
    /// </summary>
    uint32_t _Nx;

    /// <summary>
    /// Size of the layer in the y-dimension.
    /// </summary>
    uint32_t _Ny;

    /// <summary>
    /// Size of the layer in the z-dimension.
    /// </summary>
    uint32_t _Nz;

    /// <summary>
    /// Size of the layer in the w-dimension.
    /// </summary>
    uint32_t _Nw;

    /// <summary>
    /// Total number of dimensions in the layer.
    /// </summary>
    uint32_t _dimensions;

    /// <summary>
    /// Flag indicating if dimensions are provided.
    /// </summary>
    bool _bDimensionsProvided;

    /// <summary>
    /// Weight initialization method.
    /// </summary>
    WeightInitialization _weightInit;

    /// <summary>
    /// Scaling factor for weight initialization.
    /// </summary>
    float _weightInitScale;

    /// <summary>
    /// Initial bias value.
    /// </summary>
    float _biasInit;

    /// <summary>
    /// Size of the kernel in the x-dimension.
    /// </summary>
    uint32_t _kernelX;

    /// <summary>
    /// Size of the kernel in the y-dimension.
    /// </summary>
    uint32_t _kernelY;

    /// <summary>
    /// Size of the kernel in the z-dimension.
    /// </summary>
    uint32_t _kernelZ;

    /// <summary>
    /// Stride of the kernel in the x-dimension.
    /// </summary>
    uint32_t _kernelStrideX;

    /// <summary>
    /// Stride of the kernel in the y-dimension.
    /// </summary>
    uint32_t _kernelStrideY;

    /// <summary>
    /// Stride of the kernel in the z-dimension.
    /// </summary>
    uint32_t _kernelStrideZ;

    /// <summary>
    /// Padding of the kernel in the x-dimension.
    /// </summary>
    uint32_t _kernelPaddingX;

    /// <summary>
    /// Padding of the kernel in the y-dimension.
    /// </summary>
    uint32_t _kernelPaddingY;

    /// <summary>
    /// Padding of the kernel in the z-dimension.
    /// </summary>
    uint32_t _kernelPaddingZ;

    /// <summary>
    /// Total number of dimensions in the kernel.
    /// </summary>
    uint32_t _kernelDimensions;

    /// <summary>
    /// Scaling factors for Batch Normalization.
    /// </summary>
    std::vector<float> _vScaleBN;

    /// <summary>
    /// Bias factors for Batch Normalization.
    /// </summary>
    std::vector<float> _vBiasBN;

    /// <summary>
    /// Running mean values for Batch Normalization.
    /// </summary>
    std::vector<float> _vRunningMeanBN;

    /// <summary>
    /// Running variance values for Batch Normalization.
    /// </summary>
    std::vector<float> _vRunningVarianceBN;

    /// <summary>
    /// Weight normalization factor.
    /// </summary>
    float _weightNorm;

    /// <summary>
    /// Delta normalization factor.
    /// </summary>
    float _deltaNorm;

    /// <summary>
    /// Dropout probability.
    /// </summary>
    float _pDropout;

    /// <summary>
    /// Activation function.
    /// </summary>
    Activation _activation;

    /// <summary>
    /// Sparseness penalty parameter (p).
    /// </summary>
    float _sparsenessPenalty_p;

    /// <summary>
    /// Sparseness penalty parameter (beta).
    /// </summary>
    float _sparsenessPenalty_beta;

    /// <summary>
    /// Additional attributes associated with the layer.
    /// </summary>
    uint32_t _attributes;

    /// <summary>
    /// Slope parameter for the ReLU activation.
    /// </summary>
    float _RELUSlope;

    /// <summary>
    /// Alpha parameter for the ELU activation.
    /// </summary>
    float _ELUAlpha;

    /// <summary>
    /// Lambda parameter for the SELU activation.
    /// </summary>
    float _SELULambda;

    /// <summary>
    /// Default constructor for LayerDescriptor.
    /// </summary>
    LayerDescriptor();
};

struct MinMaxSpan {
    uint32_t minX;
    uint32_t maxX;
    uint32_t span;
};

bool LoadLayerDescriptorNetCDF(const std::string& fname, netCDF::NcFile& nc, uint32_t index, LayerDescriptor& ld);
std::ostream& operator<< (std::ostream& out, LayerDescriptor& d);
uint32_t MPI_Bcast_LayerDescriptor(LayerDescriptor& d);
