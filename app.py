import pickle
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum
import pandas as pd
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Dict
import random
from fastapi.exceptions import HTTPException
from sklearn.cluster import KMeans
import numpy as np

app = FastAPI()
handler = Mangum(app)

traits = pd.read_csv(r"traits/1406.csv",encoding = "cp1252")
question_set = pd.read_csv(r"question_sets/100sets_questionfile2.csv",encoding = "cp1252")
df_subtraits_desc = pd.read_csv(r"subtraits_desc.csv",encoding = "cp1252")

#############################################
from pydantic import BaseModel
from typing import Dict, List, Any

class SubTraitScore(BaseModel):
    category: str
    sub_category: str
    score: str
    clustor_category: str
    summary_response: str

class PredictReq(BaseModel):
    set_number: int
    score: Dict[str, float]

class PredictResponse(BaseModel):
    cluster: int
    # personality: str
    traits: List[str]
    overallscore: float
    candidate_desc: dict
    # candidate_overall_description: str
    # candidate_detailed_summary: dict
    highlow: dict
    ocean_desc: Dict[str, str]
    results: Dict[str, float]
    subtraits_score: list
    description_for_recruiter:dict
    # Agreeableness: Dict[str, float]
    # Conscientiousness: Dict[str, float]
    # Extraversion: Dict[str, float]
    # Neuroticism: Dict[str, float]
    # Openness: Dict[str, float]


class Question(BaseModel):
    bigfive: str
    question: str
    question_code: str

class Set(BaseModel):
    set_number: int
class QuestionResp(BaseModel):
    set_number: int
    questions: List[Question]


####################################################

# from schema import Question, QuestionResp

@app.post("/questions", response_model=QuestionResp)
async def read_questions(data:Set):
    question_resp = []
    set_list = [i for i in range(1, 101)]
    SET = random.choice(set_list)
    # SET = jsonable_encoder(data)['set_number']
    # SET = 95
    print("#######SET_NU", SET)

    setwise_question_df = question_set[question_set["SET"] == int(SET)]
    for index in setwise_question_df.index:
        question_resp.append(
            Question(
                bigfive=setwise_question_df["BigFive"][index],
                question_code=setwise_question_df["Question Code"][index],
                question=setwise_question_df["Question"][index],
            )
        )
    return QuestionResp(set_number=int(SET), questions=question_resp)


# from schema import PredictReq, PredictResponse
# from validate_score import validate_predict_req

#########################################################
# TODO: recheck

set1_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'AGR9', 'CSN1', 'CSN2', 'CSN3', 'CSN7', 'EST10', 'EST5', 'EST6', 'EST8', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN2', 'OPN5', 'OPN9']
set2_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR5', 'AGR9', 'CSN1', 'CSN3', 'CSN5', 'CSN8', 'EST10', 'EST6', 'EST8', 'EST9', 'EXT3', 'EXT6', 'EXT8', 'EXT9', 'OPN6', 'OPN8', 'OPN9']
set3_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'AGR9', 'CSN10', 'CSN3', 'CSN6', 'CSN9', 'EST2', 'EST4', 'EST5', 'EST7', 'EXT3', 'EXT5', 'EXT7', 'EXT8', 'OPN6', 'OPN8', 'OPN9']
set4_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST10', 'EST3', 'EST8', 'EST9', 'EXT10', 'EXT4', 'EXT5', 'EXT8', 'OPN1', 'OPN10', 'OPN9']
set5_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR5', 'CSN1', 'CSN10', 'CSN3', 'CSN4', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT2', 'EXT3', 'EXT5', 'EXT8', 'OPN4', 'OPN6', 'OPN7', 'OPN9']
set6_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN3', 'CSN5', 'CSN8', 'CSN9', 'EST1', 'EST10', 'EST8', 'EST9', 'EXT1', 'EXT4', 'EXT5', 'EXT8', 'OPN3', 'OPN8', 'OPN9']
set7_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN10', 'CSN2', 'CSN3', 'CSN9', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT1', 'EXT3', 'EXT5', 'EXT8', 'OPN10', 'OPN2', 'OPN9']
set8_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN2', 'CSN3', 'CSN8', 'CSN9', 'EST10', 'EST2', 'EST7', 'EST9', 'EXT3', 'EXT5', 'EXT6', 'EXT8', 'OPN1', 'OPN6', 'OPN7', 'OPN9']
set9_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN3', 'CSN5', 'CSN8', 'EST1', 'EST10', 'EST7', 'EST9', 'EXT1', 'EXT4', 'EXT5', 'EXT8', 'OPN10', 'OPN2', 'OPN7', 'OPN9']
set10_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN4', 'EST10', 'EST2', 'EST8', 'EST9', 'EXT2', 'EXT4', 'EXT5', 'EXT8', 'OPN1', 'OPN6', 'OPN9']
set11_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN1', 'CSN3', 'CSN4', 'CSN8', 'EST10', 'EST2', 'EST5', 'EST7', 'EXT3', 'EXT5', 'EXT7', 'EXT8', 'OPN1', 'OPN6', 'OPN7', 'OPN9']
set12_question_code = 	['AGR10', 'AGR3', 'AGR6', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN5', 'CSN7', 'EST2', 'EST4', 'EST5', 'EST8', 'EXT10', 'EXT3', 'EXT5', 'EXT8', 'OPN10', 'OPN4', 'OPN9']
set13_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN3', 'CSN5', 'CSN7', 'EST2', 'EST4', 'EST8', 'EST9', 'EXT4', 'EXT6', 'EXT8', 'EXT9', 'OPN3', 'OPN7', 'OPN8', 'OPN9']
set14_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN6', 'CSN8', 'EST3', 'EST4', 'EST5', 'EST7', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN9']
set15_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST10', 'EST3', 'EST5', 'EST8', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN2', 'OPN5', 'OPN7', 'OPN9']
set16_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST1', 'EST10', 'EST8', 'EST9', 'EXT1', 'EXT3', 'EXT5', 'EXT8', 'OPN6', 'OPN8', 'OPN9']
set17_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'EST10', 'EST5', 'EST6', 'EST7', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN10', 'OPN8', 'OPN9']
set18_question_code = 	['AGR10', 'AGR3', 'AGR6', 'AGR8', 'CSN2', 'CSN3', 'CSN7', 'CSN9', 'EST10', 'EST2', 'EST5', 'EST8', 'EXT2', 'EXT4', 'EXT5', 'EXT8', 'OPN10', 'OPN2', 'OPN7', 'OPN9']
set19_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'AGR9', 'CSN1', 'CSN2', 'CSN3', 'CSN7', 'EST4', 'EST6', 'EST7', 'EST9', 'EXT2', 'EXT3', 'EXT5', 'EXT8', 'OPN5', 'OPN8', 'OPN9']
set20_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'CSN2', 'CSN3', 'CSN7', 'CSN9', 'EST10', 'EST6', 'EST7', 'EST9', 'EXT3', 'EXT6', 'EXT8', 'EXT9', 'OPN1', 'OPN10', 'OPN7', 'OPN9']
set21_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN10', 'CSN3', 'CSN5', 'CSN9', 'EST10', 'EST3', 'EST7', 'EST9', 'EXT4', 'EXT6', 'EXT8', 'EXT9', 'OPN10', 'OPN8', 'OPN9']
set22_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST4', 'EST5', 'EST6', 'EST7', 'EXT4', 'EXT7', 'EXT8', 'EXT9', 'OPN3', 'OPN7', 'OPN8', 'OPN9']
set23_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST1', 'EST10', 'EST5', 'EST8', 'EXT4', 'EXT7', 'EXT8', 'EXT9', 'OPN2', 'OPN3', 'OPN9']
set24_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR5', 'CSN1', 'CSN2', 'CSN3', 'CSN7', 'EST3', 'EST4', 'EST5', 'EST7', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN10', 'OPN4', 'OPN7', 'OPN9']
set25_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN3', 'CSN4', 'CSN7', 'CSN9', 'EST10', 'EST2', 'EST7', 'EST9', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN5', 'OPN7', 'OPN8', 'OPN9']
set26_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST4', 'EST5', 'EST6', 'EST8', 'EXT1', 'EXT4', 'EXT5', 'EXT8', 'OPN3', 'OPN4', 'OPN7', 'OPN9']
set27_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST10', 'EST2', 'EST8', 'EST9', 'EXT2', 'EXT4', 'EXT5', 'EXT8', 'OPN1', 'OPN6', 'OPN9']
set28_question_code = 	['AGR10', 'AGR3', 'AGR6', 'AGR8', 'CSN10', 'CSN3', 'CSN5', 'CSN9', 'EST1', 'EST10', 'EST8', 'EST9', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN1', 'OPN10', 'OPN7', 'OPN9']
set29_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'AGR9', 'CSN3', 'CSN6', 'CSN8', 'CSN9', 'EST4', 'EST6', 'EST7', 'EST9', 'EXT3', 'EXT5', 'EXT6', 'EXT8', 'OPN1', 'OPN6', 'OPN9']
set30_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST10', 'EST3', 'EST7', 'EST9', 'EXT1', 'EXT3', 'EXT8', 'EXT9', 'OPN1', 'OPN5', 'OPN9']
set31_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN6', 'CSN8', 'EST3', 'EST4', 'EST8', 'EST9', 'EXT4', 'EXT6', 'EXT8', 'EXT9', 'OPN3', 'OPN4', 'OPN9']
set32_question_code = 	['AGR10', 'AGR3', 'AGR6', 'AGR8', 'CSN3', 'CSN5', 'CSN8', 'CSN9', 'EST1', 'EST10', 'EST5', 'EST7', 'EXT4', 'EXT7', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN7', 'OPN9']
set33_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN8', 'EST3', 'EST4', 'EST5', 'EST8', 'EXT2', 'EXT3', 'EXT5', 'EXT8', 'OPN6', 'OPN8', 'OPN9']
set34_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN2', 'CSN3', 'EST3', 'EST4', 'EST7', 'EST9', 'EXT1', 'EXT4', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN9']
set35_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN10', 'CSN3', 'CSN6', 'CSN9', 'EST1', 'EST10', 'EST7', 'EST9', 'EXT4', 'EXT6', 'EXT8', 'EXT9', 'OPN2', 'OPN6', 'OPN9']
set36_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'AGR9', 'CSN3', 'CSN5', 'CSN8', 'CSN9', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT4', 'EXT7', 'EXT8', 'EXT9', 'OPN2', 'OPN6', 'OPN9']
set37_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'AGR9', 'CSN3', 'CSN6', 'CSN8', 'CSN9', 'EST1', 'EST10', 'EST8', 'EST9', 'EXT3', 'EXT5', 'EXT7', 'EXT8', 'OPN3', 'OPN4', 'OPN9']
set38_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN3', 'CSN4', 'CSN8', 'CSN9', 'EST2', 'EST4', 'EST8', 'EST9', 'EXT10', 'EXT3', 'EXT5', 'EXT8', 'OPN1', 'OPN5', 'OPN9']
set39_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR5', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST1', 'EST4', 'EST8', 'EST9', 'EXT1', 'EXT4', 'EXT5', 'EXT8', 'OPN1', 'OPN10', 'OPN9']
set40_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'CSN2', 'CSN3', 'CSN7', 'CSN9', 'EST1', 'EST4', 'EST5', 'EST8', 'EXT2', 'EXT3', 'EXT5', 'EXT8', 'OPN2', 'OPN3', 'OPN7', 'OPN9']
set41_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST10', 'EST3', 'EST5', 'EST7', 'EXT3', 'EXT5', 'EXT6', 'EXT8', 'OPN4', 'OPN6', 'OPN7', 'OPN9']
set42_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'CSN3', 'CSN4', 'CSN7', 'CSN9', 'EST10', 'EST2', 'EST7', 'EST9', 'EXT3', 'EXT6', 'EXT8', 'EXT9', 'OPN1', 'OPN5', 'OPN7', 'OPN9']
set43_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'CSN1', 'CSN3', 'CSN6', 'CSN7', 'EST2', 'EST4', 'EST5', 'EST8', 'EXT4', 'EXT5', 'EXT6', 'EXT8', 'OPN4', 'OPN6', 'OPN7', 'OPN9']
set44_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'AGR9', 'CSN1', 'CSN3', 'CSN5', 'CSN8', 'EST10', 'EST2', 'EST7', 'EST9', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN1', 'OPN10', 'OPN9']
set45_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN8', 'EST4', 'EST6', 'EST7', 'EST9', 'EXT10', 'EXT3', 'EXT8', 'EXT9', 'OPN1', 'OPN3', 'OPN9']
set46_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST1', 'EST10', 'EST5', 'EST8', 'EXT1', 'EXT3', 'EXT8', 'EXT9', 'OPN6', 'OPN7', 'OPN8', 'OPN9']
set47_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN1', 'CSN10', 'CSN2', 'CSN3', 'EST4', 'EST5', 'EST6', 'EST8', 'EXT2', 'EXT3', 'EXT5', 'EXT8', 'OPN1', 'OPN6', 'OPN7', 'OPN9']
set48_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN6', 'CSN8', 'EST1', 'EST4', 'EST8', 'EST9', 'EXT10', 'EXT4', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN9']
set49_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN4', 'EST10', 'EST2', 'EST5', 'EST8', 'EXT10', 'EXT3', 'EXT5', 'EXT8', 'OPN3', 'OPN4', 'OPN9']
set50_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'CSN1', 'CSN10', 'CSN3', 'CSN6', 'EST1', 'EST10', 'EST5', 'EST7', 'EXT4', 'EXT6', 'EXT8', 'EXT9', 'OPN1', 'OPN10', 'OPN7', 'OPN9']
set51_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN4', 'EST1', 'EST4', 'EST8', 'EST9', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN1', 'OPN5', 'OPN9']
set52_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'AGR9', 'CSN3', 'CSN5', 'CSN8', 'CSN9', 'EST10', 'EST3', 'EST7', 'EST9', 'EXT10', 'EXT4', 'EXT5', 'EXT8', 'OPN3', 'OPN4', 'OPN9']
set53_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN8', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT1', 'EXT3', 'EXT8', 'EXT9', 'OPN1', 'OPN6', 'OPN9']
set54_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN1', 'CSN3', 'CSN5', 'CSN7', 'EST2', 'EST4', 'EST5', 'EST7', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN10', 'OPN4', 'OPN7', 'OPN9']
set55_question_code = 	['AGR10', 'AGR3', 'AGR6', 'AGR8', 'CSN1', 'CSN10', 'CSN3', 'CSN4', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT3', 'EXT6', 'EXT8', 'EXT9', 'OPN2', 'OPN6', 'OPN7', 'OPN9']
set56_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'CSN3', 'CSN6', 'CSN8', 'CSN9', 'EST4', 'EST5', 'EST6', 'EST8', 'EXT1', 'EXT3', 'EXT8', 'EXT9', 'OPN2', 'OPN5', 'OPN7', 'OPN9']
set57_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN10', 'CSN2', 'CSN3', 'EST1', 'EST10', 'EST5', 'EST7', 'EXT2', 'EXT4', 'EXT5', 'EXT8', 'OPN2', 'OPN3', 'OPN7', 'OPN9']
set58_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN10', 'CSN2', 'CSN3', 'EST10', 'EST2', 'EST5', 'EST8', 'EXT1', 'EXT3', 'EXT5', 'EXT8', 'OPN10', 'OPN2', 'OPN7', 'OPN9']
set59_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'AGR9', 'CSN1', 'CSN3', 'CSN6', 'CSN8', 'EST10', 'EST6', 'EST7', 'EST9', 'EXT4', 'EXT5', 'EXT7', 'EXT8', 'OPN2', 'OPN5', 'OPN9']
set60_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'CSN1', 'CSN3', 'CSN6', 'CSN7', 'EST10', 'EST3', 'EST5', 'EST8', 'EXT1', 'EXT3', 'EXT8', 'EXT9', 'OPN10', 'OPN4', 'OPN7', 'OPN9']
set61_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR7', 'CSN3', 'CSN4', 'CSN8', 'CSN9', 'EST1', 'EST4', 'EST8', 'EST9', 'EXT4', 'EXT5', 'EXT7', 'EXT8', 'OPN3', 'OPN7', 'OPN8', 'OPN9']
set62_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN2', 'CSN3', 'CSN8', 'CSN9', 'EST10', 'EST3', 'EST5', 'EST7', 'EXT4', 'EXT5', 'EXT7', 'EXT8', 'OPN2', 'OPN5', 'OPN9']
set63_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN3', 'CSN4', 'CSN8', 'CSN9', 'EST10', 'EST6', 'EST7', 'EST9', 'EXT4', 'EXT5', 'EXT6', 'EXT8', 'OPN10', 'OPN8', 'OPN9']
set64_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN5', 'CSN8', 'EST10', 'EST5', 'EST6', 'EST8', 'EXT10', 'EXT3', 'EXT5', 'EXT8', 'OPN1', 'OPN5', 'OPN9']
set65_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN1', 'CSN3', 'CSN6', 'CSN8', 'EST10', 'EST6', 'EST7', 'EST9', 'EXT1', 'EXT3', 'EXT5', 'EXT8', 'OPN2', 'OPN5', 'OPN7', 'OPN9']
set66_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN1', 'CSN3', 'CSN5', 'CSN7', 'EST4', 'EST5', 'EST6', 'EST8', 'EXT10', 'EXT3', 'EXT8', 'EXT9', 'OPN1', 'OPN6', 'OPN7', 'OPN9']
set67_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT1', 'EXT4', 'EXT8', 'EXT9', 'OPN1', 'OPN5', 'OPN7', 'OPN9']
set68_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN4', 'EST3', 'EST4', 'EST8', 'EST9', 'EXT4', 'EXT6', 'EXT8', 'EXT9', 'OPN3', 'OPN8', 'OPN9']
set69_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN1', 'CSN2', 'CSN3', 'CSN7', 'EST1', 'EST10', 'EST5', 'EST7', 'EXT10', 'EXT4', 'EXT8', 'EXT9', 'OPN2', 'OPN5', 'OPN7', 'OPN9']
set70_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'AGR9', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'EST3', 'EST4', 'EST5', 'EST7', 'EXT10', 'EXT4', 'EXT8', 'EXT9', 'OPN1', 'OPN3', 'OPN9']
set71_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR5', 'AGR9', 'CSN10', 'CSN2', 'CSN3', 'CSN9', 'EST3', 'EST4', 'EST5', 'EST8', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN3', 'OPN4', 'OPN9']
set72_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST4', 'EST6', 'EST7', 'EST9', 'EXT1', 'EXT3', 'EXT5', 'EXT8', 'OPN2', 'OPN5', 'OPN7', 'OPN9']
set73_question_code = 	['AGR10', 'AGR3', 'AGR5', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN6', 'CSN7', 'EST1', 'EST4', 'EST8', 'EST9', 'EXT10', 'EXT4', 'EXT5', 'EXT8', 'OPN5', 'OPN8', 'OPN9']
set74_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'AGR9', 'CSN1', 'CSN10', 'CSN2', 'CSN3', 'EST1', 'EST10', 'EST8', 'EST9', 'EXT3', 'EXT5', 'EXT7', 'EXT8', 'OPN10', 'OPN8', 'OPN9']
set75_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST1', 'EST4', 'EST5', 'EST7', 'EXT4', 'EXT5', 'EXT6', 'EXT8', 'OPN3', 'OPN7', 'OPN8', 'OPN9']
set76_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'CSN1', 'CSN3', 'CSN5', 'CSN7', 'EST3', 'EST4', 'EST8', 'EST9', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN1', 'OPN6', 'OPN7', 'OPN9']
set77_question_code = 	['AGR10', 'AGR3', 'AGR6', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN4', 'EST1', 'EST10', 'EST7', 'EST9', 'EXT4', 'EXT5', 'EXT7', 'EXT8', 'OPN2', 'OPN5', 'OPN9']
set78_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST10', 'EST2', 'EST5', 'EST8', 'EXT2', 'EXT3', 'EXT5', 'EXT8', 'OPN4', 'OPN5', 'OPN9']
set79_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'AGR9', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT3', 'EXT6', 'EXT8', 'EXT9', 'OPN10', 'OPN2', 'OPN9']
set80_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR7', 'AGR9', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'EST1', 'EST10', 'EST7', 'EST9', 'EXT1', 'EXT3', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN9']
set81_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR5', 'CSN10', 'CSN3', 'CSN4', 'CSN9', 'EST1', 'EST4', 'EST7', 'EST9', 'EXT2', 'EXT3', 'EXT8', 'EXT9', 'OPN2', 'OPN6', 'OPN7', 'OPN9']
set82_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN6', 'EST3', 'EST4', 'EST5', 'EST8', 'EXT4', 'EXT7', 'EXT8', 'EXT9', 'OPN4', 'OPN6', 'OPN9']
set83_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'CSN10', 'CSN3', 'CSN5', 'CSN9', 'EST10', 'EST2', 'EST5', 'EST7', 'EXT3', 'EXT5', 'EXT7', 'EXT8', 'OPN6', 'OPN7', 'OPN8', 'OPN9']
set84_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN3', 'CSN4', 'CSN7', 'CSN9', 'EST1', 'EST4', 'EST7', 'EST9', 'EXT10', 'EXT3', 'EXT8', 'EXT9', 'OPN4', 'OPN6', 'OPN7', 'OPN9']
set85_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR7', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST1', 'EST4', 'EST5', 'EST8', 'EXT1', 'EXT3', 'EXT8', 'EXT9', 'OPN3', 'OPN4', 'OPN9']
set86_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'AGR9', 'CSN3', 'CSN6', 'CSN8', 'CSN9', 'EST10', 'EST3', 'EST5', 'EST8', 'EXT1', 'EXT3', 'EXT5', 'EXT8', 'OPN1', 'OPN3', 'OPN9']
set87_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN3', 'CSN5', 'CSN8', 'CSN9', 'EST1', 'EST4', 'EST7', 'EST9', 'EXT3', 'EXT5', 'EXT6', 'EXT8', 'OPN10', 'OPN4', 'OPN7', 'OPN9']
set88_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'CSN10', 'CSN3', 'CSN6', 'CSN9', 'EST10', 'EST2', 'EST8', 'EST9', 'EXT1', 'EXT4', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN7', 'OPN9']
set89_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR7', 'AGR9', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'EST1', 'EST4', 'EST5', 'EST8', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN1', 'OPN10', 'OPN9']
set90_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST10', 'EST2', 'EST8', 'EST9', 'EXT3', 'EXT7', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN7', 'OPN9']
set91_question_code = 	['AGR10', 'AGR3', 'AGR7', 'AGR8', 'CSN10', 'CSN3', 'CSN4', 'CSN9', 'EST1', 'EST10', 'EST5', 'EST8', 'EXT10', 'EXT3', 'EXT5', 'EXT8', 'OPN3', 'OPN4', 'OPN7', 'OPN9']
set92_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR5', 'AGR9', 'CSN10', 'CSN3', 'CSN4', 'CSN9', 'EST1', 'EST4', 'EST7', 'EST9', 'EXT10', 'EXT3', 'EXT5', 'EXT8', 'OPN2', 'OPN6', 'OPN9']
set93_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST10', 'EST2', 'EST8', 'EST9', 'EXT2', 'EXT4', 'EXT5', 'EXT8', 'OPN3', 'OPN4', 'OPN7', 'OPN9']
set94_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'AGR9', 'CSN3', 'CSN6', 'CSN7', 'CSN9', 'EST10', 'EST5', 'EST6', 'EST8', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN4', 'OPN5', 'OPN9']
set95_question_code = 	['AGR1', 'AGR10', 'AGR2', 'AGR3', 'CSN3', 'CSN6', 'CSN8', 'CSN9', 'EST10', 'EST5', 'EST6', 'EST7', 'EXT10', 'EXT3', 'EXT8', 'EXT9', 'OPN10', 'OPN4', 'OPN7', 'OPN9']
set96_question_code = 	['AGR10', 'AGR3', 'AGR4', 'AGR8', 'CSN1', 'CSN10', 'CSN3', 'CSN6', 'EST2', 'EST4', 'EST5', 'EST7', 'EXT4', 'EXT5', 'EXT6', 'EXT8', 'OPN4', 'OPN6', 'OPN7', 'OPN9']
set97_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR6', 'AGR9', 'CSN1', 'CSN3', 'CSN4', 'CSN7', 'EST4', 'EST6', 'EST8', 'EST9', 'EXT2', 'EXT4', 'EXT5', 'EXT8', 'OPN5', 'OPN8', 'OPN9']
set98_question_code = 	['AGR1', 'AGR10', 'AGR3', 'AGR4', 'CSN1', 'CSN10', 'CSN3', 'CSN5', 'EST1', 'EST10', 'EST7', 'EST9', 'EXT2', 'EXT3', 'EXT5', 'EXT8', 'OPN1', 'OPN6', 'OPN7', 'OPN9']
set99_question_code = 	['AGR10', 'AGR3', 'AGR6', 'AGR8', 'AGR9', 'CSN1', 'CSN10', 'CSN3', 'CSN6', 'EST10', 'EST3', 'EST5', 'EST7', 'EXT2', 'EXT4', 'EXT8', 'EXT9', 'OPN4', 'OPN6', 'OPN9']
set100_question_code = 	['AGR10', 'AGR2', 'AGR3', 'AGR8', 'CSN2', 'CSN3', 'CSN7', 'CSN9', 'EST1', 'EST4', 'EST5', 'EST7', 'EXT3', 'EXT5', 'EXT7', 'EXT8', 'OPN1', 'OPN3', 'OPN7', 'OPN9']


# set4_question_code = set4_question_code.sort()
def error_string(set_number: int, question_code: str):
    return Exception("keys in dict should be in " + str(question_code) + " for set number " + str(set_number))


# from schema import PredictReq
def validate_predict_req(data: PredictReq):
    if data.set_number == 1 and list(data.score.keys()) != set1_question_code:
        raise error_string(data.set_number, str(set1_question_code))
    if data.set_number == 2 and list(data.score.keys()) != set2_question_code:
        raise error_string(data.set_number, str(set2_question_code))
    if data.set_number == 3 and list(data.score.keys()) != set3_question_code:
        raise error_string(data.set_number, str(set3_question_code))
    if data.set_number == 4 and list(data.score.keys()) != set4_question_code:
        raise error_string(data.set_number, str(set4_question_code))
    if data.set_number == 5 and list(data.score.keys()) != set5_question_code:
        raise error_string(data.set_number, str(set5_question_code))
    if data.set_number == 6 and list(data.score.keys()) != set6_question_code:
        raise error_string(data.set_number, str(set6_question_code))
    if data.set_number == 7 and list(data.score.keys()) != set7_question_code:
        raise error_string(data.set_number, str(set7_question_code))
    if data.set_number == 8 and list(data.score.keys()) != set8_question_code:
        raise error_string(data.set_number, str(set8_question_code))
    if data.set_number == 9 and list(data.score.keys()) != set9_question_code:
        raise error_string(data.set_number, str(set9_question_code))
    if data.set_number == 10 and list(data.score.keys()) != set10_question_code:
        raise error_string(data.set_number, str(set10_question_code))
    if data.set_number == 11 and list(data.score.keys()) != set11_question_code:
        raise error_string(data.set_number, str(set11_question_code))
    if data.set_number == 12 and list(data.score.keys()) != set12_question_code:
        raise error_string(data.set_number, str(set12_question_code))
    if data.set_number == 13 and list(data.score.keys()) != set13_question_code:
        raise error_string(data.set_number, str(set13_question_code))
    if data.set_number == 14 and list(data.score.keys()) != set14_question_code:
        raise error_string(data.set_number, str(set14_question_code))
    if data.set_number == 15 and list(data.score.keys()) != set15_question_code:
        raise error_string(data.set_number, str(set15_question_code))
    if data.set_number == 16 and list(data.score.keys()) != set16_question_code:
        raise error_string(data.set_number, str(set16_question_code))
    if data.set_number == 17 and list(data.score.keys()) != set17_question_code:
        raise error_string(data.set_number, str(set17_question_code))
    if data.set_number == 18 and list(data.score.keys()) != set18_question_code:
        raise error_string(data.set_number, str(set18_question_code))
    if data.set_number == 19 and list(data.score.keys()) != set19_question_code:
        raise error_string(data.set_number, str(set19_question_code))
    if data.set_number == 20 and list(data.score.keys()) != set20_question_code:
        raise error_string(data.set_number, str(set20_question_code))

    if data.set_number == 21 and list(data.score.keys()) != set21_question_code:
        raise error_string(data.set_number, str(set21_question_code))
    if data.set_number == 22 and list(data.score.keys()) != set22_question_code:
        raise error_string(data.set_number, str(set22_question_code))
    if data.set_number == 23 and list(data.score.keys()) != set23_question_code:
        raise error_string(data.set_number, str(set23_question_code))
    if data.set_number == 24 and list(data.score.keys()) != set24_question_code:
        raise error_string(data.set_number, str(set24_question_code))
    if data.set_number == 25 and list(data.score.keys()) != set25_question_code:
        raise error_string(data.set_number, str(set25_question_code))
    if data.set_number == 26 and list(data.score.keys()) != set26_question_code:
        raise error_string(data.set_number, str(set26_question_code))
    if data.set_number == 27 and list(data.score.keys()) != set27_question_code:
        raise error_string(data.set_number, str(set27_question_code))
    if data.set_number == 28 and list(data.score.keys()) != set28_question_code:
        raise error_string(data.set_number, str(set28_question_code))
    if data.set_number == 29 and list(data.score.keys()) != set29_question_code:
        raise error_string(data.set_number, str(set29_question_code))
    if data.set_number == 30 and list(data.score.keys()) != set30_question_code:
        raise error_string(data.set_number, str(set30_question_code))

    if data.set_number == 31 and list(data.score.keys()) != set31_question_code:
        raise error_string(data.set_number, str(set31_question_code))
    if data.set_number == 32 and list(data.score.keys()) != set32_question_code:
        raise error_string(data.set_number, str(set32_question_code))
    if data.set_number == 33 and list(data.score.keys()) != set33_question_code:
        raise error_string(data.set_number, str(set33_question_code))
    if data.set_number == 34 and list(data.score.keys()) != set34_question_code:
        raise error_string(data.set_number, str(set34_question_code))
    if data.set_number == 35 and list(data.score.keys()) != set35_question_code:
        raise error_string(data.set_number, str(set35_question_code))
    if data.set_number == 36 and list(data.score.keys()) != set36_question_code:
        raise error_string(data.set_number, str(set36_question_code))
    if data.set_number == 37 and list(data.score.keys()) != set37_question_code:
        raise error_string(data.set_number, str(set37_question_code))
    if data.set_number == 38 and list(data.score.keys()) != set38_question_code:
        raise error_string(data.set_number, str(set38_question_code))
    if data.set_number == 39 and list(data.score.keys()) != set39_question_code:
        raise error_string(data.set_number, str(set39_question_code))
    if data.set_number == 40 and list(data.score.keys()) != set40_question_code:
        raise error_string(data.set_number, str(set40_question_code))

    if data.set_number == 41 and list(data.score.keys()) != set41_question_code:
        raise error_string(data.set_number, str(set41_question_code))
    if data.set_number == 42 and list(data.score.keys()) != set42_question_code:
        raise error_string(data.set_number, str(set42_question_code))
    if data.set_number == 43 and list(data.score.keys()) != set43_question_code:
        raise error_string(data.set_number, str(set43_question_code))
    if data.set_number == 44 and list(data.score.keys()) != set44_question_code:
        raise error_string(data.set_number, str(set44_question_code))
    if data.set_number == 45 and list(data.score.keys()) != set45_question_code:
        raise error_string(data.set_number, str(set45_question_code))
    if data.set_number == 46 and list(data.score.keys()) != set46_question_code:
        raise error_string(data.set_number, str(set46_question_code))
    if data.set_number == 47 and list(data.score.keys()) != set47_question_code:
        raise error_string(data.set_number, str(set47_question_code))
    if data.set_number == 48 and list(data.score.keys()) != set48_question_code:
        raise error_string(data.set_number, str(set48_question_code))
    if data.set_number == 49 and list(data.score.keys()) != set49_question_code:
        raise error_string(data.set_number, str(set49_question_code))
    if data.set_number == 50 and list(data.score.keys()) != set50_question_code:
        raise error_string(data.set_number, str(set50_question_code))

    if data.set_number == 51 and list(data.score.keys()) != set51_question_code:
        raise error_string(data.set_number, str(set51_question_code))
    if data.set_number == 52 and list(data.score.keys()) != set52_question_code:
        raise error_string(data.set_number, str(set52_question_code))
    if data.set_number == 53 and list(data.score.keys()) != set53_question_code:
        raise error_string(data.set_number, str(set53_question_code))
    if data.set_number == 54 and list(data.score.keys()) != set54_question_code:
        raise error_string(data.set_number, str(set54_question_code))
    if data.set_number == 55 and list(data.score.keys()) != set55_question_code:
        raise error_string(data.set_number, str(set55_question_code))
    if data.set_number == 56 and list(data.score.keys()) != set56_question_code:
        raise error_string(data.set_number, str(set56_question_code))
    if data.set_number == 57 and list(data.score.keys()) != set57_question_code:
        raise error_string(data.set_number, str(set57_question_code))
    if data.set_number == 58 and list(data.score.keys()) != set58_question_code:
        raise error_string(data.set_number, str(set58_question_code))
    if data.set_number == 59 and list(data.score.keys()) != set59_question_code:
        raise error_string(data.set_number, str(set59_question_code))
    if data.set_number == 60 and list(data.score.keys()) != set60_question_code:
        raise error_string(data.set_number, str(set60_question_code))

    if data.set_number == 61 and list(data.score.keys()) != set61_question_code:
        raise error_string(data.set_number, str(set61_question_code))
    if data.set_number == 62 and list(data.score.keys()) != set62_question_code:
        raise error_string(data.set_number, str(set62_question_code))
    if data.set_number == 63 and list(data.score.keys()) != set63_question_code:
        raise error_string(data.set_number, str(set63_question_code))
    if data.set_number == 64 and list(data.score.keys()) != set64_question_code:
        raise error_string(data.set_number, str(set64_question_code))
    if data.set_number == 65 and list(data.score.keys()) != set65_question_code:
        raise error_string(data.set_number, str(set65_question_code))
    if data.set_number == 66 and list(data.score.keys()) != set66_question_code:
        raise error_string(data.set_number, str(set66_question_code))
    if data.set_number == 67 and list(data.score.keys()) != set67_question_code:
        raise error_string(data.set_number, str(set67_question_code))
    if data.set_number == 68 and list(data.score.keys()) != set68_question_code:
        raise error_string(data.set_number, str(set68_question_code))
    if data.set_number == 69 and list(data.score.keys()) != set69_question_code:
        raise error_string(data.set_number, str(set69_question_code))
    if data.set_number == 70 and list(data.score.keys()) != set70_question_code:
        raise error_string(data.set_number, str(set70_question_code))

    if data.set_number == 71 and list(data.score.keys()) != set71_question_code:
        raise error_string(data.set_number, str(set71_question_code))
    if data.set_number == 72 and list(data.score.keys()) != set72_question_code:
        raise error_string(data.set_number, str(set72_question_code))
    if data.set_number == 73 and list(data.score.keys()) != set73_question_code:
        raise error_string(data.set_number, str(set73_question_code))
    if data.set_number == 74 and list(data.score.keys()) != set74_question_code:
        raise error_string(data.set_number, str(set74_question_code))
    if data.set_number == 75 and list(data.score.keys()) != set75_question_code:
        raise error_string(data.set_number, str(set75_question_code))
    if data.set_number == 76 and list(data.score.keys()) != set76_question_code:
        raise error_string(data.set_number, str(set76_question_code))
    if data.set_number == 77 and list(data.score.keys()) != set77_question_code:
        raise error_string(data.set_number, str(set77_question_code))
    if data.set_number == 78 and list(data.score.keys()) != set78_question_code:
        raise error_string(data.set_number, str(set78_question_code))
    if data.set_number == 79 and list(data.score.keys()) != set79_question_code:
        raise error_string(data.set_number, str(set79_question_code))
    if data.set_number == 80 and list(data.score.keys()) != set80_question_code:
        raise error_string(data.set_number, str(set80_question_code))

    if data.set_number == 81 and list(data.score.keys()) != set81_question_code:
        raise error_string(data.set_number, str(set81_question_code))
    if data.set_number == 82 and list(data.score.keys()) != set82_question_code:
        raise error_string(data.set_number, str(set82_question_code))
    if data.set_number == 83 and list(data.score.keys()) != set83_question_code:
        raise error_string(data.set_number, str(set83_question_code))
    if data.set_number == 84 and list(data.score.keys()) != set84_question_code:
        raise error_string(data.set_number, str(set84_question_code))
    if data.set_number == 85 and list(data.score.keys()) != set85_question_code:
        raise error_string(data.set_number, str(set85_question_code))
    if data.set_number == 86 and list(data.score.keys()) != set86_question_code:
        raise error_string(data.set_number, str(set86_question_code))
    if data.set_number == 87 and list(data.score.keys()) != set87_question_code:
        raise error_string(data.set_number, str(set87_question_code))
    if data.set_number == 88 and list(data.score.keys()) != set88_question_code:
        raise error_string(data.set_number, str(set88_question_code))
    if data.set_number == 89 and list(data.score.keys()) != set89_question_code:
        raise error_string(data.set_number, str(set89_question_code))
    if data.set_number == 90 and list(data.score.keys()) != set90_question_code:
        raise error_string(data.set_number, str(set90_question_code))

    if data.set_number == 91 and list(data.score.keys()) != set91_question_code:
        raise error_string(data.set_number, str(set91_question_code))
    if data.set_number == 92 and list(data.score.keys()) != set92_question_code:
        raise error_string(data.set_number, str(set92_question_code))
    if data.set_number == 93 and list(data.score.keys()) != set93_question_code:
        raise error_string(data.set_number, str(set93_question_code))
    if data.set_number == 94 and list(data.score.keys()) != set94_question_code:
        raise error_string(data.set_number, str(set94_question_code))
    if data.set_number == 95 and list(data.score.keys()) != set95_question_code:
        raise error_string(data.set_number, str(set95_question_code))
    if data.set_number == 96 and list(data.score.keys()) != set96_question_code:
        raise error_string(data.set_number, str(set96_question_code))
    if data.set_number == 97 and list(data.score.keys()) != set97_question_code:
        raise error_string(data.set_number, str(set97_question_code))
    if data.set_number == 98 and list(data.score.keys()) != set98_question_code:
        raise error_string(data.set_number, str(set98_question_code))
    if data.set_number == 99 and list(data.score.keys()) != set99_question_code:
        raise error_string(data.set_number, str(set99_question_code))
    if data.set_number == 100 and list(data.score.keys()) != set100_question_code:
        raise error_string(data.set_number, str(set100_question_code))


##################################################################
import ast
import json


def cluster(X):
    k_means = KMeans(n_clusters=3).fit(X)
    return X.groupby(k_means.labels_)\
            .transform('mean').sum(1)\
            .rank(method='dense').sub(1)\
            .astype(int).to_frame()


@app.post("/predict", response_model=PredictResponse)
def predict_cluster(data: PredictReq):
    """
    Score will be question code and values
    """
    try:
        validate_predict_req(data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    print("##########PREDICT  API######")
    df = pd.DataFrame([data.score], columns=data.score.keys())
    print("dataframe created", df)
    # Load the saved model
    model_file = str(data.set_number) + "pp.pkl"
    model = "models/" + model_file
    with open(model, "rb") as file:
        classifier = pickle.load(file)
    print("Loaded model", model)
    print(type(classifier))
    df = df.reindex(sorted(df.columns), axis=1)
    y_pred = classifier.predict(df)
    my_cluster = y_pred[0]
    print("My Cluster", my_cluster)
    my_df = traits[traits["SET"] == data.set_number]
    my_df = my_df[my_df["Clusters"] == my_cluster]
    my_df = my_df[
        [
            "SET",
            "Clusters",
            "Extroversion",
            "Neurotic",
            "Agreeable",
            "Conscientious",
            "Openness",
            "Label",
            "Traits",
            'overall_description',
            'HighLow',
            'detailed_summary',
            "strengths",
            "weaknesses",
            "improvements",
            'OCEAN_desc', 'overall_desc_to_recruiter',
       'strengths_to_recruiter', 'weekness_to_recruiter',
       'improvements_to_recruiter'
        ]
    ]
    personality = my_df["Label"].to_string(index=False)
    my_df = my_df.reset_index(drop=True)
    overall_score = (
                            my_df["Openness"]
                            + my_df["Conscientious"]
                            + my_df["Extroversion"]
                            + my_df["Agreeable"]
                            + my_df["Neurotic"]
                    ) / 5

    # overallscorediscription = "".join()

    my_sums = pd.DataFrame()
    my_sums["Extroversion"] = my_df["Extroversion"]
    my_sums["Agreeable"] = my_df["Agreeable"]
    my_sums["Conscientious"] = my_df["Conscientious"]
    my_sums["Openness"] = my_df["Openness"]
    my_sums["Neurotic"] = my_df["Neurotic"]
    my_sums = my_sums[
        ["Agreeable", "Conscientious", "Openness", "Extroversion", "Neurotic"]
    ]
    my_results = my_sums.T
    my_results = my_results.reset_index().rename(
        columns={"index": "Personality", int(0): "Score"}
    )
    my_results = my_results.set_index("Personality")["Score"].to_dict()
    print("@@@@@@@@@@MYRESULTS", my_results)
    mytraits = " ".join(str(i) for i in my_df["Traits"])
    mytraits = ast.literal_eval(mytraits)

    overallscorediscription = "".join(str(i) for i in list(my_df["overall_description"]))
    print("#########overallscorediscription", overallscorediscription)
    detailedsummary = "".join(str(i) for i in list(my_df["detailed_summary"]))
    detailedsummary_separated_fields = my_df[["overall_description", "strengths",
            "weaknesses",
            "improvements"]].to_dict('records')[0]

    detailedsummary_to_recruiter_separated_fields = my_df[["overall_desc_to_recruiter",
                                              "strengths_to_recruiter",
                                              "weekness_to_recruiter", 'improvements_to_recruiter']].to_dict('records')[0]
    print("##########detailedsummary", detailedsummary)
    high_low = my_df["HighLow"].to_dict()
    high_low = ast.literal_eval(high_low[0])
    oceandesc = my_df["OCEAN_desc"].to_dict()
    oceandesc = ast.literal_eval(oceandesc[0])

    #######Sub Traits Scoring########
    df_ocean_sub_categories = pd.read_csv(r"ocean_sub_categories2.csv", encoding="cp1252")
    df_payload = pd.DataFrame([data.score], columns=data.score.keys())
    df_payload = df_payload.reindex(sorted(df_payload.columns), axis=1)
    df_payload = df_payload.T.reset_index().rename(
        columns={"index": "Question Code", int(0): "Score"})
    df_payload['sub category'] = df_payload["Question Code"].replace(
        df_ocean_sub_categories.set_index('Question Code')['sub category'].to_dict())
    df_payload['+/-keyed'] = df_payload["Question Code"].replace(df_ocean_sub_categories.set_index('Question Code')['+/- keyed'].to_dict())
    df_payload['Category'] = df_payload["Question Code"].replace(df_ocean_sub_categories.set_index('Question Code')['Category'].to_dict())
    df_payload['Score'] = df_payload['Score']

    df_plus = pd.DataFrame()
    df_plus = df_payload[df_payload['+/-keyed'] == 'plus'][
        ['Question Code', 'Score', 'sub category', '+/-keyed', 'Category']].copy()
    remapvalues_for_minus = {20: 100, 40: 80, 80: 40, 100: 20}
    df_minus = pd.DataFrame()
    df_payload["Updated_Scores_for_minus"] = df_payload[df_payload['+/-keyed'] == 'minus']['Score'].replace(
        remapvalues_for_minus)
    df_payload = df_payload.dropna()
    df_minus = df_payload.copy()
    df_minus = df_minus[['Question Code', 'Updated_Scores_for_minus', 'sub category', '+/-keyed', 'Category']]
    df_minus.rename(columns={'Updated_Scores_for_minus': 'Score'}, inplace=True)

    frames = [df_plus, df_minus]
    result = pd.concat(frames)
    result = result.rename(columns={"Category":"category"})
    # result_main = result.copy()
    # result_main = result_main.groupby(['Category'])['Score'].sum() * 20
    # result_main = result_main.to_dict()
    result_subtraits = result.groupby(['category', 'sub category'])['Score'].sum() * 20
    result_subtraits = result_subtraits.to_frame().rename(columns={'Score': 'subtraits scores'}).reset_index()

    ########
    subtraits_count = list()

    for i in result_subtraits["category"].unique().tolist():
        subtraits_count.append(
            result_subtraits[result_subtraits["category"] == str(i)].groupby("category")["sub category"].count().to_frame().rename(
                columns={'sub category': 'subtraits_counts'}).reset_index().to_dict('r')[0])

    df_subtraits_count = pd.DataFrame(subtraits_count)
    df_subtraits_count['subtraits_counts'] = df_subtraits_count['subtraits_counts']
    df_subtraits_count["outof"] = 100 / df_subtraits_count['subtraits_counts']
    df_subtraits_count

    result_subtraits = pd.merge(result_subtraits, df_subtraits_count, on=["category"])
    ########
    print("1905@@@@@@@@@@@@@@@@@", result_subtraits.columns.tolist())
    # result_subtraits = result_subtraits.to_json(orient="records")
    import numpy as np
    conditions2 = [
        (result_subtraits['subtraits scores'] < 10),
        (result_subtraits['subtraits scores'] >= 10) & (result_subtraits['subtraits scores'] < 15),
        (result_subtraits['subtraits scores'] >= 15)
    ]

    values2 = ['Low', 'Neutral', 'High']
    result_subtraits['cluster_category'] = np.select(conditions2, values2)

    # cols = ['Subtraits Scores']
    # mapping = {0: 'Low', 1: 'Neutral', 2: 'High'}
    # print("BEFORE CLUSTER")
    # result_subtraits['Cluster_id'] = result_subtraits.groupby('Category')[cols].apply(cluster)
    # print("AFTER CLUSTER")
    # result_subtraits['cluster_category'] = result_subtraits['Cluster_id'].map(mapping)

    result_subtrait = pd.merge(result_subtraits, df_subtraits_desc, on=["sub category", "cluster_category"])
    result_subtrait = result_subtrait[['category', 'sub category', 'subtraits scores', "outof", 'cluster_category', 'response']].to_dict('records')
    # Subtrait_details = {"Subtraits_details": result_subtrait[
    #     ['Category', 'sub category', 'Subtraits Scores', 'cluster_category', 'response']].to_dict('records')}
    # Subtrait_details
    # print("10051005", result_subtrait['Category'] )
    #
    # category = result_subtrait['Category'].to_string(index=False)
    # sub_category =  result_subtrait['sub category'].to_string(index=False)
    # score =  result_subtrait["Subtraits Scores"].to_string(index=False)
    # clustor_category = result_subtrait['cluster_category'].to_string(index=False)
    # summary_response = result_subtrait['Response'].to_string(index=False)

    # agreeableness  = result_subtraits[result_subtraits['Category'] == 'Agreeableness'].set_index("Category")["Score"].to_dict()
    # print("%%%%%",agreeableness)
    # conscientiousness = result_subtraits[result_subtraits['Category'] == 'Conscientiousness'].set_index("Category")["Score"].to_dict()
    # extraversion = result_subtraits[result_subtraits['Category'] == 'Extraversion'].set_index("Category")["Score"].to_dict()
    # neuroticism = result_subtraits[result_subtraits['Category'] == 'Neuroticism'].set_index("Category")["Score"].to_dict()
    # openness = result_subtraits[result_subtraits['Category'] == 'Openness'].set_index("Category")["Score"].to_dict()

    print("SCORES CALCULATED SUCCESSFULLY")
    # print("@@@@@@@@@", result_main)
    print("&&&&&&&&&&", result_subtraits)

    result_main = result.groupby(['category'])['Score'].sum() * 20
    print("May 1805", result_main)
    result_main = result_main.to_frame().rename(columns={'Score': 'Trait Scores'}).reset_index()
    dict = {"Agreeableness" : 'agreeable', "Conscientiousness" : 'conscientious', "Neuroticism": 'neurotic','Extraversion':'extraversion',"Openness":"openness"}

    result_main = result_main.replace({"category": dict})
    ########################
    import numpy as np
    conditions1 = [
        (result_main['Trait Scores'] < 40),
        (result_main['Trait Scores'] >= 40) & (result_main['Trait Scores'] < 60),
        (result_main['Trait Scores'] >= 60) & (result_main['Trait Scores'] <= 100)
    ]
    values1 = ['Low', 'Neutral', 'High']

    result_main['HighLow'] = np.select(conditions1, values1)

    highlow_values = result_main[['category', 'HighLow']].set_index("category")["HighLow"]
    highlow_values = highlow_values.to_dict()
    over_score = result_main["Trait Scores"].sum()*0.2
    result_main = result_main.set_index("category")["Trait Scores"].to_dict()

    return PredictResponse(
        cluster=my_cluster,
        # personality=personality,
        traits=mytraits,
        overallscore=over_score,
        candidate_desc=detailedsummary_separated_fields,
        highlow=highlow_values,
        ocean_desc=oceandesc,
        results=result_main,
        subtraits_score=result_subtrait,
        description_for_recruiter=detailedsummary_to_recruiter_separated_fields)

    #     Agreeableness=agreeableness,
    # Conscientiousness=conscientiousness,
    # Extraversion= extraversion,
    # Neuroticism=neuroticism ,
    # Openness=openness

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999, reload=True)


