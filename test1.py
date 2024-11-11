import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline ,BitsAndBytesConfig
import pandas as pd
import numpy as np


#few shots + prompt 
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

Answer: (C)"""
        
    },
    {
        'role': 'user',
        'content':"""질문 : 이차방정식 x2 - 5x + 3 = 0의 두 근이 log5α와 log5β일 때, logαβ + logβα의 값은?\"
        선택지 : 
        A : 17/3
        B : 19/3
        C : 21/5
        D : 23/5
        """
    },
    {
        'role': 'assistant',
        'content': """“Question: If the roots of the quadratic equation x2 - 5x + 3 = 0 are log5α and log5β, what is the value of logαβ + logβα?

        Options:
        A : 17/3
        B : 19/3
        C : 21/5
        D : 23/5

Let's think through it step by step. We will refer to Wikipedia for mathematical information. The problem is to find the value of \( \log \alpha \beta + \log \beta \alpha \) given that the roots of the quadratic equation \( x^2 - 5x + 3 = 0 \) are \( \log_5 \alpha \) and \( \log_5 \beta \). According to Vieta's formulas for the roots of a quadratic equation, the sum of the roots is \(-\frac{b}{a}\) and the product of the roots is \(\frac{c}{a}\). Here, \(a\), \(b\), and \(c\) are the coefficients of the quadratic equation. For the given equation \( x^2 - 5x + 3 = 0 \), we have \(a = 1\), \(b = -5\), and \(c = 3\), so the sum of the roots is \(5\) and the product of the roots is \(3\). Since \( \log_5 \alpha \) and \( \log_5 \beta \) are the roots, the following conditions hold: 1. \( \log_5 \alpha + \log_5 \beta = 5 \) 2. \( \log_5 \alpha \cdot \log_5 \beta = 3 \). By the properties of logarithms, \( \log \alpha \beta + \log \beta \alpha \) can be expressed as \( \log \alpha^2 \beta^2 \), which is \( \log (\alpha \beta)^2 \). Since \( \alpha \beta = 5^3 \) (from the product of the roots), this is \( \log 5^6 \), which equals \( 6 \cdot \log 5 \). Since \( \log 5 \) is 1, this simplifies to 6. However, considering additional operations and quadratic expansions involving \( \log_5 \alpha \) and \( \log_5 \beta \), the final simplified result in fractional form is \( \frac{19}{3} \). Therefore, the answer is (B).

Answer: (B)"""
        
    },
    {
        'role': 'user',
        'content':"""질문 : (가)에 대한 설명으로 옳지 않은 것은? (가)은/는 의병계열과 애국계몽 운동 계열의 비밀결사가 모여 결성된 조직으로,
      총사령 박상진을 중심으로 독립군 양성을 목적으로 하였다.\"
        선택지 : 
        A : 북간도에서 무장 독립 단체인 중광단을 조직하였다.
        B : 군자금을 모집하고 친일파를 공격하였다.
        C : 공화제 국가 수립을 지향하였다.
        D : 경상도 일대에서 결성되어 전국 조직으로 확대하였다.
        """
    },
    {
        'role': 'assistant',
        'content': """
        Question: Which of the following statements about (가) is NOT correct? (가) is an organization formed by a secret alliance of the righteous army faction and the patriotic enlightenment movement faction, with General Park Sang-jin at its center, aimed at training independence fighters.

        Options:
        A : It organized the armed independence group Junggwangdan in North Manchuria.
        B : It raised military funds and attacked pro-Japanese figures.
        C : It aimed to establish a republic.
        D : It was formed in the Gyeongsang-do region and expanded into a nationwide organization.

Let's think through it step by step. According to Wikipedia for Korean history, the Daehan Gwangbokhoe (대한광복회) was an independence movement organization formed on July 15, 1915, in Dalseong Park, Daegu, by the unification of the righteous army faction and the enlightenment movement faction. This organization aimed to restore national sovereignty and achieve independence by establishing a military academy in Manchuria to train independence fighters and prepare for war against Japan when military strength was sufficient. It also planned to relocate Koreans to Manchuria, develop farmland, and supply food and troops. The Daehan Gwangbokhoe sought to establish a republic, reflecting the widespread republican ideals among independence activists following the advocacy of republicanism by the Sinminhoe. The organization raised military funds and attacked pro-Japanese figures, using commercial organizations disguised as contact points across the country and setting up lodgings and general stores in Dandong and Changchun as contact bases. However, the Daehan Gwangbokhoe did not organize the armed independence group Junggwangdan in North Manchuria. Junggwangdan was an armed struggle organization formed by Daechongyo (Dangun) groups led by Na Cheol and O Gi-ho, and it was the predecessor of the Bukro Gunjeongseo established after the March 1st Movement. The Bukro Gunjeongseo is famous for the unit led by General Kim Jwa-jin in the Battle of Cheongsanri. Therefore, the incorrect statement about (가) is (A). The Daehan Gwangbokhoe did not organize Junggwangdan in North Manchuria, as it was a separate organization. Thus, the answer is (A).

Answer: (A)"""
    },

    {'role': 'user', 'content': f"{content}"}
] 
    
    return messages
#creating inputs
def create_contents(df):
    container = []
    for q,c in zip(df['문제'], df['선택지']):
        container.append(q + ' ' + c)
    id = df['id']
    
    return id , container

# generate 하고 save 까지 해야함 
def generate(model,tokenizer,id,content):
    
    data_dict={'id':[],'output':[]}
    
    pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    ) 

    generation_args = { 
        "max_new_tokens": 1000, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample":  False, 
    }    

    for i, cont in zip(id,content):

        model_input = create_input(cont)  #model_input = [ {K:str}]
        output = pipe(model_input, **generation_args) 
        print(output[0]['generated_text'])
        data_dict['id'].append(i)
        data_dict['output'].append(output[0]['generated_text'])

    return data_dict

def make_csv_of_output(output:dict):
    result = pd.DataFrame(output)

    return result


if __name__ == '__main__':
    # model and tokenizer
    device ='cuda:0'
    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )



    torch.random.manual_seed(0) 
    model = AutoModelForCausalLM.from_pretrained( 
        "Phi-3-mini-128k-instruct",  
        device_map='auto',  
        quantization_config =nf4_config,
        trust_remote_code=True,  
    ) 

    tokenizer = AutoTokenizer.from_pretrained("Phi-3-mini-128k-instruct",
                                              quantization_config=nf4_config)

    model.eval()
    #dataset
    df= pd.read_csv("data/test_data.csv")

    df1= df.iloc[:250]
    df2 =df.iloc[250:500]
    df3 =df.iloc[500:750]
    df4 =df.iloc[750:]

    id , content = create_contents(df1)
    
    data_dict = generate(model,tokenizer,id,content)

    result = make_csv_of_output(data_dict)

    result.to_csv('result1.csv',index=False)

        
