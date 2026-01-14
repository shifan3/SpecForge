FROM cuda12.9-cp311

RUN apt update && apt install -y git libnuma1 libnuma-dev
RUN git clone https://github.com/shifan3/SpecForge.git -b support-qwen3-vl
WORKDIR /SpecForge
RUN python3.11 -m pip install setuptools -U
RUN python3.11 -m pip install -e .