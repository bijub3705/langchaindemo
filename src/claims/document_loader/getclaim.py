import csv
import json
from src.claims.document_loader.summarize import getModel, summarize
from langchain_core.documents import Document

def getClaimData(file_name):
    claim_file=file_name
    claim_list=[]
    chat_model, messages=getModel()
    with open(claim_file,mode="r") as f:
        lines=list(csv.reader(f))
    i=0
    for line in lines:
        i+=1
        if i==1: continue
        #print(line)
        #print("Claim ")
        #print(line)
        #print(i)
        
        claim_summary=summarize(chat_model, messages,line).split("\n")[0]
        #print(claim_summary)
        line.append(claim_summary)
        claim_list.append(Document(page_content=claim_summary, metadata={"mem_id":line[0],"Claim_no":line[1]}))
        
        
        #print("end")
    return claim_list
if __name__=="__main__":
    claim_list=getClaimData("C:\\Users\\prade\\python\\projectllm\\bijulangchain\\langchaindemo\\docs\\context_help.csv")
    
    #print(claim_list)
        
