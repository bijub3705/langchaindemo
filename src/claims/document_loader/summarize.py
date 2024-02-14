from src.claims.utils import get_constants
import os
import json
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate



#json_data={"claim_id":"F123","member": "Sam Brady","provider": "Memorial Hospital","service":"Wellness Exam","net_amount":"$250","copay":"$0"}

#print(json.dumps(json_data))


prompt = """
input:MEM3,H1234,Vincent Joe,1/3/2024,Prov2,St Lukes,11-Outpatient,Shoulder injury,ultra sound,Paid,320,300,250,0,50,none,none
output: This entry represents a medical claim for a patient named Vincent Joe, with a member ID of MEM3 and claim ID of H1234. The claim was for services rendered on January 3, 2024. The medical provider, identified as St. Lukes (Prov2), administered an ultrasound procedure for the diagnosis of a shoulder injury. The claim status is marked as "Paid." The billed amount for the services was $320, with an allowed amount of $300. The paid amount was $250, and there was no co-pay. The deductible for this claim was $50. The denied reason and hold reason fields are both specified as "none," indicating that there were no reasons for denial or hold on the claim.

input:MEM1,Q1236,John Doe,1/3/2024,PROV3,Baptist,11-Outdetient,Elbow dislocation,ultra sound,Denied,100,100,0,0,0,annual limit reached,none
output: This entry represents a medical claim for a patient named John Doe, with a member ID of MEM1 and claim ID of Q1236. The claim was for services rendered on January 3, 2024. The medical provider, identified as Baptist (PROV3), administered an ultrasound procedure for the diagnosis of elbow dislocation. However, the claim status is marked as "Denied." The billed amount for the services was $100, with an allowed amount of $100. The paid amount, co-pay, and deductible fields are all $0, indicating that no payment or financial transactions occurred. The reason for denial is specified as "annual limit reached," indicating that the claim was denied because it exceeded the annual limit specified by the insurance policy. The hold reason field is specified as "none," indicating that there were no reasons for placing the claim on hold
"""



def getModel():
    config=get_constants().MODELS.get("summarize-model")
    url=config.api_url #  "https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud"
    token= config.api_token
    llm = HuggingFaceEndpoint(
        endpoint_url=url,
        huggingfacehub_api_token=token,
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    
    messages = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt),
        HumanMessagePromptTemplate.from_template("{message}")
    ])
    return chat_model, messages
def summarize(chat_model, messages, user_input):
    
    chain = messages | chat_model
    result = chain.invoke({"message": user_input}).content
    
    return result

if __name__=="__main__":
    model, messages=getModel()
    print(summarize(model, messages,"[MEM1,Q1234,John Doe,1/1/2024,PROV1,Mayo Clinic,11-Outpatient,Elbow dislocation,ultra sound,Paid,150,100,80,20,0,,]"))
