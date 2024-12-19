# Dialogue_Summarization | `2024.05.13 - 2024.05.27`
[NLP / LLM] 한국어 일상 대화 요약

<br>

## Summary

### 🛠️ 목표

현대 사회에서 다양한 플랫폼과 메신저를 통해 생성되는 대화 데이터의 양은 기하급수적으로 증가하고 있다. 이러한 대화 데이터는 고객 응대, 소셜 네트워킹, 업무 커뮤니케이션 등 다양한 분야에서 중요한 정보를 포함하지만, 대화의 비정형적, 휘발적 특성 때문에 핵심 내용을 빠르고 정확하게 파악하는 것이 어렵다. <br>

다양한 일상 주제를 포함한 한국어 일상 대화 데이터를 효과적으로 요약할 수 있는 자연어 처리 모델을 만드는 것이 프로젝트의 목표이다. <br>

### ⚙️ 수행 역할

- **데이터 오류 수정 및 데이터 증강(AEDA)**: 영어 대화문을 한국어로 번역한 데이터셋이기에 오타나 불완전한 문장이 많았기에 데이터를 정제하고 오류를 수정하여 모델 학습의 질을 개선

- **다양한 LLM 모델을 활용한 실험**

- **QLoRA, Quantization 경량화 적용**


### 📈 결과 및 직무에 적용할 점

1. 일상 대화 요약 뿐만 아니라 다양한 NLP 및 LLM 기반 프로젝트에 적용할 수 있는 기술적 기반 마련
2. 데이터 증강 경험을 통해 자연어 데이터에 대한 품질 개선 및 다양성 확보 방법론 습득
3. 다양한 경량화 기법들을 활용하여 저자원 환경에서도 모델 효율성 극대화

<br>

## 프로젝트 상세

### 📝 데이터 설명

- 데이터 형태: .csv 파일
- 데이터 개수: train (`12,457`), dev (`499`), test(`250`), hidden-test(`249`)

- 다양한 주제를 가진 대화문과 이에 대한 요약문을 포함하고 있다.
- 최소 2턴, 최대 60턴으로 대화가 구성된다.

![image](https://github.com/user-attachments/assets/122a0022-7ea6-4006-9a2e-7001cd52afb2)

> - fname: 대화 고유 번호
> - dialogue: 최소 2명에서 최대 7명이 등장하여 나누는 대화 내용. 각 발화자를 구분하기 위해 special token #Person"N"#을 사용하며, 발화자의 대화가 끝나면 \n으로 구분한다.
> - summary: 해당 대화를 바탕으로 작성된 요약문

### 📊 EDA

<p align="center">
  <img src="https://github.com/user-attachments/assets/42e9e00a-f516-4971-915b-ee7be49acceb" alt="image" />
  <img src="https://github.com/user-attachments/assets/24e9f656-7658-4d1a-80da-1bbcd09cf9c0" alt="image" />
</p>

- 요약문은 대화문 길이의 20%로 요약된다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a840ffaa-760b-4680-9e90-314e46fd7815" alt="image" width="50%"/>
  <img src="https://github.com/user-attachments/assets/ddd26001-85a8-4d79-bd01-b53533f146eb" alt="image" />
</p>

<span style="color:#808080"> 대화문 데이터 오류는 [여기](https://fish-target-b61.notion.site/NLP-Competition-5a07a68bee894418a559a2dcc2f238f0)를 참조해주세요. </span>

<br>

### 🔍 데이터 전처리
 
- 데이터 오류 정리본은 다음과 같은 과정을 통해 정제하였다.

```python

# 'dialogue' 컬럼에 대한 전처리
df['dialogue'] = df['dialogue'].apply(lambda x: x.replace('ㅇ로', '으로'))
df['dialogue'] = df['dialogue'].apply(lambda x: x.replace('제ㅏ', '제가'))
df['dialogue'] = df['dialogue'].apply(lambda x: x.replace('ㅍ알', ' 알'))
df['dialogue'] = df['dialogue'].apply(lambda x: x.replace('ㄷ거', '거'))

# 'dialogue' 컬럼에 대한 문자열 대체
df['dialogue'] = df['dialogue'].str.replace('##', '#')
df['dialogue'] = df['dialogue'].str.replace('#', '0')
df['dialogue'] = df['dialogue'].str.replace('#', '#Person2#:')
df['dialogue'] = df['dialogue'].str.replace('#', '#Person1#:')
...
```

- 데이터셋이 일상대화를 담은 만큼 개인정보가 포함되어 있어 이를 마스킹하여 제공되었다. 따라서, 대화문 및 요약문에서 마스킹된 값들을 tokenizer에 special token으로 포함시켰다. <br>

```python

# 특수 토큰 정의
special_tokens = ['#PassportNumber#', '#CardNumber#', '#Person3#', '#DateOfBirth#', '#Address#', '#CarNumber#', '#Email#',
                  '#Person2#', '#Person6#', '#Person1#', '#Person#', '#Person7#', '#Person5#', '#PhoneNumber#', '#Person4#', '#SSN#']


```

### 🧠 모델링

 - 📐 평가지표 : Rouge F1 Score

 ![image](https://github.com/user-attachments/assets/1fcc8604-3222-4d86-b78e-c52a5f41eee3)

 - **프롬프트**

   - 인코더 모델 `BERT` : 입력 텍스트 중심의 task 정의, \<s> 토큰으로 시작과 종료를 알린다.
  
      ```python
      def train_prompt_roberta(row):

        topic = row['topic'] if 'topic' in row else ''

        prompt = f"""<s> 다음의 대화를 요약하세요. 아래의 지침을 따르세요:

        1. 대화 길이의 20% 이내로 요약
        2. 대화 내에서 중요하게 명명된 개체의 이름을 그대로 사용
        3. 은어나 약어 없이 공식적으로 사용하는 언어로 작성

        주제: {topic}
        대화: {row['dialogue']}
        요약: {row['summary']} </s>"""

        return prompt
      ```

   - 디코더 모델 `LLaMA` : 시스템 메세지 및 어시스턴트 응답을 활용하여 명확히 구분한다.

      ```python
      def train_prompt_llama(row):
          topic = row['topic'] if 'topic' in row else ''

          prompt = f"""<s> <|im_start|>system
          주제를 참고해 다음 대화를 요약하세요. 아래의 지침을 따르세요:

          1. 대화 길이의 20% 이내로 요약
          2. 대화 내에서 중요하게 명명된 개체의 이름을 그대로 사용
          3. 은어나 약어 없이 공식적으로 사용하는 언어로 작성

          주제: {topic}
          대화: {row['dialogue']}
          <|im_end|>
          <|im_start|>assistant
          요약: {row['summary']}
          <|im_end|> </s>"""

          return prompt
      ```

   - 인코더-디코더 모델 `T5` : task 이름을 명시적으로 포함하고, Seq2Seq 형식으로 작성한다.

      ```python
      def train_prompt_t5(row):
          topic = row['topic'] if 'topic' in row else ''

          prompt = f"""summarize dialogue:
          topic: {topic}
          dialogue: {row['dialogue']}

          summary: {row['summary']}"""

          return prompt
      ```

 - **모델 구성**

    사전 학습에 한국어 데이터셋을 사용한 모델들을 불러와 fine-tuning을 진행하였다. <br>

    | 모델 `BERT`                   | Score     | ROUGE-1     | ROUGE-2     | ROUGE-L      |
    |:----------------------------:|:---------:|:-----------:|:-----------:|:------------:|
    | kykim/bert-kor-base          | Public    | 0.4331      | 0.2866      | 0.3375       |
    |                              | Private   | 0.4009      | 0.2738      | 0.3001       |

    | 모델 `LLaMA`               | Score     | ROUGE-1     | ROUGE-2     | ROUGE-L      |
    |:-------------------------:|:---------:|:-----------:|:-----------:|:------------:|
    | beomi/Llama-3-Open-Ko-8B  | Public    | 0.4956      | 0.3072      | 0.4094       |
    |                           | Private   | 0.5021      | 0.3014      | 0.4112       |

    | 모델 `T5`                               | Score          | ROUGE-1     | ROUGE-2     | ROUGE-L      |
    |:--------------------------------------:|:--------------:|:-----------:|:-----------:|:------------:|
    | eenzeenee/t5-base-korean-summarization | Public         | 0.4769      | 0.2755      | 0.3949       |
    |                                        | Private        | 0.4755      | 0.2637      | 0.3836       |

    | 모델 `T5`                           | Score          | ROUGE-1     | ROUGE-2     | ROUGE-L      |
    |:----------------------------------:|:--------------:|:-----------:|:-----------:|:------------:|
    | lcw99/t5-large-korean-text-summary | Public         | 0.5340      | 0.3350      | 0.4397       |
    |                                    | Private        | 0.5173      | 0.3024      | 0.4155       |

<br>

## 🎯 결과 및 기대효과

### 결과 요약

  - 본 프로젝트는 **Summarization**을 목표로 진행되었으며, Transformer의 인코더, 디코더, 인코더-디코더 구조 모델을 비교 평가하였다.

  - LLaMA 모델보다 적은 파라미터를 가진 T5 모델이 가장 높은 지표를 보여주며, 대화 요약 작업에서 인코더-디코더 구조의 강점을 입증하였다.

### 기대효과

  - **응용 가능성**: 대화 요약 기술은 고객 상담 요약, 의료 기록 요약, 회의록 작성 등의 다양한 분야에 적용 가능하다.

  - **지속적인 개선 여지**: 성능 한계를 보완하기 위한 데이터 증강, 파인튜닝, 하이퍼파라미터 조정을 통해 성능을 더욱 높일 가능성이 존재한다.

### 한계점 및 아쉬운 점

1. **데이터 정제**  
   - 영문 데이터를 단순히 한국어로 번역하였기에, 다소 어색한 번역체로 구성되어있는 데이터셋이였다. 이를 해결하여 더 좋은 데이터로 모델을 학습시킬 수 있었는데, 시간 상의 이유로 못한 것이 아쉽다.

2. **자원 제약 내에서의 효율성 부족**  
   - 경계선을 잘 정하여 그에 맞는 모델을 실험했어야 했는데, 그것이 잘 안되어 엄청난 시간 낭비로 이어졌다. 제한된 환경에 맞게 최선의 모델을 채택하고, 최선의 성능을 끌어올리는 것이 중요한 역량이라는 것을 깨닫는 시간이었다.
