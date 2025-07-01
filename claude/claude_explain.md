# Apple Game ML Solver 설계 및 구현 설명

## 1. 프로젝트 개요

### 1.1 목표
기존의 규칙 기반 솔버와 달리, **머신러닝/딥러닝**을 활용하여 사과게임(Apple Game)을 해결하는 AI를 개발하는 것이 목표입니다. 로직을 직접 프로그래밍하지 않고, AI가 스스로 최적의 전략을 학습하도록 설계했습니다.

### 1.2 ML 접근법을 선택한 이유
- **적응성**: 다양한 보드 상황에 유연하게 대응
- **최적화**: 수많은 시행착오를 통해 인간이 발견하지 못한 패턴 학습 가능
- **확장성**: 게임 규칙이 변경되어도 재훈련으로 적응 가능
- **학습 목적**: ML의 핵심인 "직접 로직을 짜지 않고 학습으로 해결"하는 방식 구현

## 2. 기술적 설계 결정

### 2.1 강화학습(Reinforcement Learning) 선택
**왜 강화학습인가?**
- 사과게임은 **순차적 의사결정** 문제입니다
- 각 행동(사과 선택)이 미래 상태에 영향을 미침
- 최종 목표(최고 점수)를 위한 장기적 전략이 필요
- 환경과의 상호작용을 통해 학습하는 구조가 적합

### 2.2 DQN (Deep Q-Network) 알고리즘 선택
**DQN을 선택한 이유:**
1. **복잡한 상태공간**: 17x10 격자의 다양한 숫자 조합
2. **연속적이지 않은 행동공간**: 사과 조합 선택은 이산적
3. **안정성**: Target Network로 학습 안정화
4. **경험 재생**: 과거 경험을 재활용하여 효율적 학습

## 3. 코드 구조 분석

### 3.1 AppleGameEnvironment 클래스
```python
class AppleGameEnvironment:
    def __init__(self, rows=10, cols=17):
        self.rows = rows
        self.cols = cols
        self.reset()
```

**설계 철학:**
- OpenAI Gym 스타일의 환경 구현
- 게임의 물리적 제약사항을 정확히 모델링
- 상태, 행동, 보상을 명확히 정의

**핵심 메서드:**
1. `reset()`: 새로운 게임 시작, 랜덤 보드 생성
2. `get_state()`: 현재 보드 상태를 ML 모델이 이해할 수 있는 형태로 변환
3. `get_valid_combinations()`: 게임 규칙에 맞는 유효한 행동들을 생성
4. `step()`: 행동을 실행하고 새로운 상태, 보상, 종료 여부 반환

**상태 표현의 혁신:**
```python
features = np.stack([
    visible_board,           # 현재 보이는 숫자들
    self.cleared.astype(float),  # 이미 제거된 위치
    np.ones_like(visible_board) * (self.moves_made / self.max_moves)  # 시간 정보
], axis=0)
```
- 3채널로 구성하여 CNN이 처리할 수 있도록 설계
- 공간적 정보와 시간적 정보를 모두 포함

### 3.2 직사각형 선택 규칙 구현
```python
def can_select_rectangle(self, positions):
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if not self.cleared[r, c] and (r, c) not in positions:
                return False
    return True
```

**설계 의도:**
- 게임의 핵심 규칙인 "직사각형 선택"을 정확히 구현
- 빈 공간은 무시하고 숫자만 선택할 수 있는 규칙 반영
- 복잡한 기하학적 제약을 단순한 알고리즘으로 해결

### 3.3 DQN 신경망 아키텍처
```python
class DQN(nn.Module):
    def __init__(self, input_channels=3, board_height=10, board_width=17):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, board_height * board_width)
```

**아키텍처 선택 이유:**

1. **Convolutional Layers**: 
   - 보드의 공간적 패턴 인식
   - 지역적 특징(숫자 조합) 추출
   - 위치 불변성 제공

2. **Progressive Feature Maps**:
   - 32 → 64 → 128로 점진적 확장
   - 저수준 특징에서 고수준 특징으로 추상화

3. **Fully Connected Layers**:
   - 전역적 의사결정을 위한 정보 통합
   - Dropout으로 과적합 방지

4. **Output Design**:
   - 각 보드 위치별 Q-value 출력
   - 170개 위치에 대한 가치 함수 학습

### 3.4 행동 선택 전략
```python
def act(self, env, state):
    valid_moves = env.get_valid_combinations()
    
    if np.random.random() <= self.epsilon:
        return random.choice(valid_moves)  # 탐험
    
    # 각 조합의 Q-value 합을 계산하여 최적 행동 선택
    for combination in valid_moves:
        score = sum(q_values[0][r * env.cols + c].item() for r, c in combination)
        if score > best_score:
            best_score = score
            best_combination = combination
```

**설계 혁신:**
- **ε-greedy 전략**: 탐험과 활용의 균형
- **조합별 점수 계산**: 개별 위치의 Q-value를 합산하여 조합 평가
- **유효한 행동만 고려**: 게임 규칙 위반 방지

## 4. 학습 과정 설계

### 4.1 보상 체계
```python
if sum(values) != 10:
    return self.get_state(), -5, False, {}  # 잘못된 조합에 큰 패널티

reward = len(action_positions)  # 제거한 사과 개수만큼 보상

if done and np.sum(~self.cleared) == 0:
    reward += 50  # 완전 클리어 보너스
```

**보상 설계 철학:**
- **즉시 보상**: 올바른 행동에 대한 즉각적 피드백
- **장기 보상**: 게임 완료 시 추가 보너스
- **패널티**: 잘못된 행동 억제

### 4.2 경험 재생과 안정화
```python
class DQNAgent:
    def __init__(self, ...):
        self.memory = deque(maxlen=10000)  # 경험 저장소
        self.update_target_every = 100     # 타겟 네트워크 업데이트 주기
        self.learn_every = 4               # 학습 주기
```

**안정화 기법:**
1. **Experience Replay**: 과거 경험 재사용으로 학습 효율성 증대
2. **Target Network**: 학습 목표의 안정성 확보
3. **Batch Learning**: 미니배치로 분산 감소

### 4.3 하이퍼파라미터 튜닝
```python
lr=0.001,           # 학습률: 안정적 학습
gamma=0.95,         # 할인률: 미래 보상 고려
epsilon=1.0,        # 초기 탐험률: 충분한 탐험
epsilon_decay=0.995, # 탐험률 감소: 점진적 활용 증가
epsilon_min=0.01    # 최소 탐험률: 지속적 탐험 보장
```

## 5. 구현상의 혁신점

### 5.1 게임 규칙의 정확한 모델링
- 직사각형 선택 제약을 수학적으로 정확히 구현
- 빈 공간 무시 규칙을 알고리즘에 반영
- 실제 게임과 동일한 환경 제공

### 5.2 효율적인 행동 공간 설계
- 모든 가능한 조합(2^170)을 고려하는 대신, 유효한 조합만 생성
- 휴리스틱으로 검색 공간 축소 (최대 20개 위치까지 고려)
- 실시간 유효성 검사로 잘못된 행동 방지

### 5.3 멀티채널 상태 표현
- 현재 보드 상태, 제거된 위치, 시간 정보를 분리하여 표현
- CNN이 각 정보를 독립적으로 처리할 수 있도록 설계
- 시간적 맥락을 공간적 정보와 통합

## 6. 예상되는 학습 패턴

### 6.1 초기 단계 (Episode 0-100)
- 랜덤한 행동으로 환경 탐험
- 기본적인 "합이 10인 조합" 학습
- 높은 탐험률로 다양한 전략 시도

### 6.2 중간 단계 (Episode 100-300)
- 지역적 최적화 패턴 학습
- 간단한 조합 (9-1, 8-2 등) 우선 선택
- 보드 상태에 따른 조건부 전략 개발

### 6.3 후기 단계 (Episode 300-500)
- 전역적 최적화 전략 학습
- "라인 오브 사이트" 전략 자동 발견
- 복잡한 조합을 활용한 고득점 달성

## 7. 기존 접근법과의 차이점

### 7.1 규칙 기반 솔버
- **규칙 기반**: 프로그래머가 전략을 직접 코딩
- **ML 기반**: AI가 스스로 전략을 발견하고 학습

### 7.2 탐욕적 알고리즘
- **탐욕적**: 현재 최선의 선택만 고려
- **강화학습**: 장기적 결과를 고려한 의사결정

### 7.3 완전 탐색
- **완전 탐색**: 모든 경우의 수 검토 (계산 복잡도 높음)
- **학습 기반**: 경험을 통해 좋은 전략만 선별적 학습

## 8. 확장 가능성

### 8.1 알고리즘 개선
- **PPO (Proximal Policy Optimization)**: 더 안정적인 정책 기반 학습
- **A3C (Asynchronous Actor-Critic)**: 병렬 학습으로 속도 향상
- **Rainbow DQN**: 다양한 DQN 개선사항 통합

### 8.2 모델 아키텍처 개선
- **ResNet**: 더 깊은 네트워크로 복잡한 패턴 학습
- **Attention Mechanism**: 중요한 보드 영역에 집중
- **Graph Neural Network**: 사과 간의 관계를 그래프로 모델링

### 8.3 훈련 최적화
- **Curriculum Learning**: 쉬운 보드부터 어려운 보드로 점진적 학습
- **Transfer Learning**: 유사한 퍼즐 게임에서 사전 훈련
- **Multi-task Learning**: 다른 목표 함수와 동시 학습

## 9. 결론

이 ML 솔버는 사과게임의 복잡한 규칙과 전략적 요소를 모두 고려하여 설계되었습니다. 강화학습의 핵심 아이디어인 "환경과의 상호작용을 통한 학습"을 충실히 구현하였으며, 게임의 물리적 제약사항을 정확히 모델링했습니다.

**핵심 혁신:**
1. **규칙 기반이 아닌 학습 기반 접근**
2. **공간적 패턴 인식을 위한 CNN 활용**
3. **장기적 전략을 위한 강화학습 적용**
4. **게임 고유 제약사항의 정확한 구현**

이 솔버는 단순히 게임을 해결하는 것을 넘어서, ML/DL이 복잡한 의사결정 문제를 어떻게 해결할 수 있는지를 보여주는 사례가 될 것입니다.