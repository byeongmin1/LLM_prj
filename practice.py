import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from vllm import LLM, SamplingParams
import pandas as pd

sampling_params = SamplingParams(temperature=0.0, 
                                 top_p=0.95,
                                 max_tokens=1024)

gpu_memory_utilization = 0.9

model_name='Qwen2-72B-Instruct-GPTQ-Int4'

def create_input(content):    
    messages = [ 
    {"role": "system", "content": """
you are a highly knowledgeable AI assistant with expertise in all fields, and you should provide responses to requests in the most detailed and friendly manner possible. You need to translate questions coming in Korean into English, think through the problem step-by-step, and provide answers accordingly. Analyze the information provided by users carefully to quickly understand their intentions and generate responses that are ethically appropriate, following norms to protect children.
                  
Start your answer with the sentence "Let's think through it step by step." and end the sentence with "Answer: ()" where the correct letter is included in the parentheses. Carefully examine the given question and options, and if necessary, refer to relevant Wikipedia entries in that field. Provide a detailed, logical explanation of the solution, and make sure to choose one letter among options A, B, C, or D. Even if none of the answers seem correct, you must select one of the letters A, B, C, or D..
                  """}, 
   {
        'role': 'user',
        'content':"""질문 : 수정 전 잔액시산표의 차변 합계액은 ￦1,000,000이다. 보험료 미경과액 ￦30,000과 이자수익 미수액 ￦20,000을 계상한 후의 수정 후 잔액시산표 차변 합계액은 얼마인가?\"
        선택지 : 
        A : ￦970,000
        B : ￦990,000
        C : ￦1,020,000
        D : ￦1,050,000
        """
    },
    {
        'role': 'assistant',
        'content': """Question: The debit total of the unadjusted trial balance is ￦1,000,000. What will be the debit total of the adjusted trial balance after accounting for the unearned insurance premium of ￦30,000 and the accrued interest income of ￦20,000?

        Options:
        A : ￦970,000
        B : ￦990,000
        C : ￦1,020,000
        D : ￦1,050,000

        Let's think through it step by step. We will refer to accounting information from Wikipedia. The given problem requires distinguishing between factors that affect the total of the adjusted trial balance and those that do not. In this case, an unearned insurance premium and accrued interest income have been added. The unearned insurance premium decreases the insurance expense (debit) while increasing the prepaid insurance (asset) by the same amount on the debit side, so it has no effect. Next, the accrued interest income of ￦20,000 represents interest that has not yet been received. This accrued interest is recorded as a debit in the trial balance. Adding the accrued interest income of ￦20,000 to the original debit total of ￦1,000,000 results in an adjusted debit total of ￦1,020,000. Therefore, the correct answer is (C).

Answer: (C)"""},
        
   

    {'role': 'user', 'content': f"{content}"}
] 
    return messages



llm = LLM(model=model_name,
            tokenizer=model_name,
            tensor_parallel_size=4, 
            pipeline_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            #cpu_offload_gb=4,
            dtype='float16',
            enforce_eager=True,
            quantization='gptq',
            seed=42,
            max_model_len = 4096,
            trust_remote_code=True,
            load_format='auto',
            )



# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]

# outputs = llm.generate(prompts, sampling_params)
# # Print the outputs
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")




df = pd.read_csv('data/train_data.csv')

data_dict = {'question' : df['문제'], 'choices' : df['선택지'] }

data = {'question': []}
for i,j in zip(data_dict['question'], data_dict['choices']):
    q =  i + ' ' + j
    result = create_input(q)
    #data['question'].append(q)
    data['question'].append(result)


for que in data['question']:
    outputs=llm.chat(que, sampling_params=sampling_params, use_tqdm = True)
    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text}")



