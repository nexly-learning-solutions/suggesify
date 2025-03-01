
FROM mcr.microsoft.com/windows/servercore:ltsc2019 AS devel

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]


WORKDIR "C:\\\\workspace"


COPY setup_env.ps1 C:\\workspace\\setup_env.ps1
RUN C:\workspace\setup_env.ps1 -skipTRT -skipCUDNN
RUN Remove-Item "C:\workspace\setup_env.ps1" -Force


COPY setup_build_env.ps1 C:\\workspace\\setup_build_env.ps1
RUN C:\workspace\setup_build_env.ps1 -TRTPath 'C:\\workspace'
RUN Remove-Item "C:\workspace\setup_build_env.ps1" -Force

RUN [Environment]::SetEnvironmentVariable('Path', $Env:Path + ';C:\Program Files\CMake\bin', [EnvironmentVariableTarget]::Machine)



RUN Invoke-WebRequest -Uri https://ftp.nluug.nl/pub/vim/pc/gvim90.exe \
    -OutFile "install_vim.exe"; \
    Start-Process install_vim.exe -Wait -ArgumentList '/S'; \
    Remove-Item install_vim.exe -Force ; \
    [Environment]::SetEnvironmentVariable('Path', $Env:Path + ';C:\Program Files (x86)\Vim\vim90', [EnvironmentVariableTarget]::Machine)


ENV chocolateyVersion=1.4.0

RUN Set-ExecutionPolicy Bypass -Scope Process -Force; \
 [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; \
 iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))


RUN choco install git -y

RUN Invoke-WebRequest -Uri https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe \
    -OutFile cuda_11.8.0_windows_network.exe; \
    Invoke-WebRequest -Uri https://7-zip.org/a/7zr.exe \
    -OutFile 7zr.exe

RUN .\7zr.exe e -i!'nsight_nvtx\nsight_nvtx\NVIDIA NVTX Installer.x86_64.Release.v1.21018621.Win64.msi' cuda_11.8.0_windows_network.exe ;

RUN cmd.exe /S /C "msiexec.exe /i 'NVIDIA NVTX Installer.x86_64.Release.v1.21018621.Win64.msi' /norestart /quiet"

RUN Remove-Item 'NVIDIA NVTX Installer.x86_64.Release.v1.21018621.Win64.msi' -Force ; \
    Remove-Item 7zr.exe -Force ; \
    Remove-Item cuda_11.8.0_windows_network.exe -Force


ENTRYPOINT ["C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\Tools\\VsDevCmd.bat", "-arch=amd64", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]

