# Use a base image with CUDA support
FROM nvidia/cuda:11.0-base

# Install PyTorch and PyTorch Lightning
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu110/torch_stable.html
RUN pip3 install pytorch-lightning

# Set the working directory
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Command to run when the container starts
CMD ["python3", "your_script.py"]

# docker build -t pytorch-lightning-gpu .
# docker run --gpus all -it pytorch-lightning-gpu