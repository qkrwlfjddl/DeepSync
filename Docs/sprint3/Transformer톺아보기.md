# Transformer architecture
![image](https://github.com/devkade/DeepSync/assets/11837072/4720863d-1975-481d-a739-50743e2ca0da)
![image](https://github.com/devkade/DeepSync/assets/11837072/e640dd2e-cddb-42eb-acf2-b1edf2f99045)

   
# Operation analysis
![image](https://github.com/devkade/DeepSync/assets/11837072/249be589-f844-49c4-a363-618af4a061e2)

## Attention
![image](https://github.com/devkade/DeepSync/assets/11837072/b5404c4f-82e4-4756-8272-8876d2a569d1)

## Feed Forward Neural Network
![image](https://github.com/devkade/DeepSync/assets/11837072/62f034cf-0cdd-4cf6-a599-e9030f753686)


# Analysis by pass
## Forward Pass
## Backward Pass

# Analysis by task
## Training
- forward + backward
## Inference
- forward
### Prefill
- inference 중에서 input length의 token으로 1개의 token을 generation 하는 과정
### Decode
- prefill 이후 eos가 나올 때 까지 n개의 output token을 generation 하는 과정
   
# Optimization scheme
## Activation Recomputation
## KV-cache
![image](https://github.com/devkade/DeepSync/assets/11837072/a40fc8ac-b5c0-44ee-b36d-a3dadbbb0028)


# DGX 구성
- A100 80G x 8
- Infiniband network x 8
