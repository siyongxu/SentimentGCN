from konlpy.tag import Okt

def read_data(filename):
    with open(filename, 'r', encoding = 'utf8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        head = data[0]
        data = data[1:]
    return head, data

dataset = 'test'
head, train_data = read_data('data/ratings_'+dataset+'.txt')

head = '\t'.join(head)
file=open('data/ratings_'+dataset+'_tokenized.txt','w', encoding = 'utf8')
file.write(head + '\n')
okt = Okt()
reviews = []
#tokenized_reviews = []
for review in train_data :
    tokenized_review = []
    words = okt.pos(review[1], norm=True, stem=True) #review를 토크나이즈
    sentence = []
    for word in words :
        sentence.append(word[0]) #토크나이즈 된 값 list에 저장 ==> string으로 변환 예정
    sentence = ' '.join(sentence) # 각 단어간 공백을 추가하여 string으로 변환 ==> TF-IDF 사용
    review[1] = sentence
    review = '\t'.join(review)
    file.write(review + '\n')

file.close()
