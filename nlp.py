import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer



def load_csvfile():
    #load csv
    csv_path =r"C:\Users\HP\Desktop\stevens\595\companies\csv" 
    os.chdir(csv_path)
    fileName=os.listdir(csv_path)
    df=pd.DataFrame()
    for i in range(len(fileName)):
        csv=pd.read_csv(fileName[i])
        df=df.append(csv)
    return df


def get_txt(filename):
    data=[]
    with open (filename,mode="r")as f:#0,1,3,5
        for i in f:
            if i!='\n':
                data.append(i.strip("\n"))
        data=[[data[0+2*i],data[1+2*i]]for i in range(len(data)//2)]
    return data

def load_txtfile():
#load txt
    txt_path=r"C:\Users\HP\Desktop\stevens\595\companies\txt"
    os.chdir(txt_path)
    fileName=os.listdir(txt_path)
    all_data=[]
    all_data.extend(get_txt(fileName[0]))
    all_data.extend(get_txt(fileName[1]))
    all_data.extend(get_txt(fileName[3]))
    all_data.extend(get_txt(fileName[5]))

    with open (fileName[2],mode="r")as f:#2
        data=[]
        for i in f:
            if i!='\n':
                data.append(i.strip("\n"))
        data=[[data[i],data[i+50]] for i in range(len(data)//2)]
        all_data.extend(data)
    
    with open (fileName[4],mode="r")as f:#4
        data=[]
        for i in f:
            if i!='\n':
                data.append(i.strip("\n"))
        data=[i.split("Purpose") for i in data]
        all_data.extend(data)
    with open (fileName[6],mode="r")as f:#6
        data=[]
        for i in f:
            if i!='\n':
                data.append(i.strip("\n"))
        data=data[0].split("],")
        data=[i.split("Purpose") for i in data]  
        all_data.extend(data)
    df_1=pd.DataFrame(all_data,columns=["Name","Purpose"])
    return df_1

       

stemmer = PorterStemmer()
def stems(text):
    return stemmer.stem(text)

lemmatizer = WordNetLemmatizer()
def lemmas(text):
    return lemmatizer.lemmatize(text)

analyser = SentimentIntensityAnalyzer()
def sentiment_analysis(text):
    return analyser.polarity_scores(text)["compound"]


def main():
    df=load_csvfile()
    df_1=load_txtfile()
    # merge data

    df=df.append(df_1)
    #analysis 
    all_scores=[]
    for i in range(len(df)):
        text=df["Purpose"].values[i]
        text=stems(text)
        text=lemmas(text)
        score=sentiment_analysis(text)
        all_scores.append(score)
    df=df.join(pd.DataFrame(all_scores,columns=["Scores"]))
    # save dataframe to csv 
    df=df.sort_values("Scores")
    df=df.drop(columns=["Unnamed: 0"])
    df.to_csv("result.csv",index=None)
    
if __name__=="__main__":
    main()
    
    