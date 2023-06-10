import pandas as pd
import numpy as np


def calc_lev_distance(a, b):
    #  레벤슈타인 거리 계산 함수 
    
    if a == b: return 0 # 같으면 0을 반환
    a_len = len(a) # a 길이
    b_len = len(b) # b 길이
    if a == "": return b_len
    if b == "": return a_len

    # 2차원 표 (a_len+1, b_len+1) 준비하기 
    matrix = [[] for i in range(a_len+1)] # 리스트 컴프리헨션을 사용하여 1차원 초기화
    for i in range(a_len+1): # 0으로 초기화
        matrix[i] = [0 for j in range(b_len+1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
    # 0일 때 초깃값을 설정
    for i in range(a_len+1):
        matrix[i][0] = i
    for j in range(b_len+1):
        matrix[0][j] = j
    # 표 채우기 

    for i in range(1, a_len+1):
        ac = a[i-1]
        for j in range(1, b_len+1):
            bc = b[j-1] 
            cost = 0 if (ac == bc) else 1  #  파이썬 조건 표현식 예:) result = value1 if condition else value2
            matrix[i][j] = min([
                matrix[i-1][j] + 1,     # 문자 제거: 위쪽에서 +1
                matrix[i][j-1] + 1,     # 문자 삽입: 왼쪽 수에서 +1   
                matrix[i-1][j-1] + cost # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
            ])
    return matrix[a_len][b_len]


class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)

    def load_data(self, filepath):
        data = pd.read_csv(filepath)    # csv 파일 read  
        questions = data['Q'].tolist()  # csv 파일에서 질문열을 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()    # csv 파일에서 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers       # questions, answer를 반납 

    def find_best_answer(self, input_sentence):  
        # Levenshtein ㄱㅓ리 기준으로, 사용자가 입력한 질문에 가장 근접한 질문을
        # questions로부터 탖음
        # 이후 이에 대한 답을 answers 중에서 인덱싱해서, 반환함 
        
        distances = [ calc_lev_distance(input_sentence, question) for question in self.questions]
            # questions 컬럼의 각 question에 대해, 사용쟈가 입력한 input_sentence 대비
            # Levenshtein 거리를 구함
            # python list comprehension을 활용하여, 반복 수행 및 list가 리턴됨 
        
        best_match_index = np.argmin(distances) # 거리값이므로,  가장 작은 인덱스를 반환 
            
        return self.answers[best_match_index] 
            # input_sentenceㅇㅘ questions간 거리가 가장 짧아지는 경우의 index를 찾아서
            # 해당 경우의 answer를 답변으로 반환함 

# CSV 파일명 및 경로 
filepath = 'ChatbotData.csv'

# 챗봇 인스턴스 생성
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복함 
while True:
    input_sentence = input('You: ') # 사용자의 질문을 입력함 
    if input_sentence.lower() == '종료': # 사용자가 '종료'를 입력한 경ㅇ, while 구문을 빠져나감 
        break
    response = chatbot.find_best_answer(input_sentence) 
        # 생성된 chatbot instance에 대해, 사용자가 입력한 input_sentence 대비, 
        # Levenshtein 거리가 가장 가까운 질문을 찾아서, 해당 질문에 대한 답변을 리턴받음 
    print('Chatbot:', response)
    
