# AI_UWB

## 설명

- 목적
    - Feature extractor가 각 파형의 중요한 부분을 캡쳐할 수 있도록 데이터를 구성
    - 각 데이터의 dimension , stepsize가 일치 해야함 → 만약 일치하지 않을 경우 따로 전처리를 해야해서 귀찮음
    - 목적: 아래 figure 와 같이 양질의 Test 데이터 만들기
    
    - 데이터 측정방법:
        - 해당 모델이 파형의 중요한 부분 feature를 캡쳐했는지 , 단지 amplitude 를 근거로 모델을 분류했는지 확인하기위해서 좀더 많은 데이터로 분류를 진행하기로 함
        - 아래 환경과 같이 측정
    





        <img src="https://github.com/kycl0131/AI_UWB/assets/79360166/2f64ed7d-0661-4ecc-aa33-bdfeb187cd26" width="400" height="300"/>

    
    해당 사진은 Moving target 1개의 (1,33,1528) 의 데이터를 한 플롯에 나타낸것
    
    timestep = 33, dim =1528
    
    ![timestep and dim](https://github.com/kycl0131/AI_UWB/assets/79360166/936ae54b-1a75-4fb6-8317-479e14882115)
    
![folder img](https://github.com/kycl0131/AI_UWB/assets/79360166/3f08f1e8-1205-4fc2-b80e-9466afec7c4e)
![property](https://github.com/kycl0131/AI_UWB/assets/79360166/10b78931-bdb8-41c8-92f0-6f067fa51d88)

위 데이터 한개는 다음과 같음

![onedata](https://github.com/kycl0131/AI_UWB/assets/79360166/dbb205e2-4e92-4e47-a809-41c2bb85ca7c)


### DataPicker 코딩중

1. Target Range 범위에서 임계값 이상인 존재인 index 검출 후

33*1528(기존 train data 크기) 박스안에 들어오게 설정

2. Background 제거

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f08466b-e034-4f99-813b-1aac37130e4b/image9.png

Target Range 범위에서 임계값 이상인 존재인 idx 검출

33*1528 박스안에 들어오게 설정

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/922a1649-7ceb-4722-ae46-9c712faca224/image10.png

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e9b86e16-9840-48b6-8239-faba730e84f7/image11.png

idx 검출후 Background 제거

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/180e896a-bcb9-461e-bad3-21683383e6f9/image12.png

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9735a2d4-5436-4d88-8a41-ff29c405699e/image13.png

Background 제거가능

신호 검출 – 검출 직전 파형

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9758f198-9d28-4136-aded-5bb818e9fa4e/image14.png

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5443142c-e174-4581-80fc-73f1adffad1d/image15.png

33 step background 제거

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/73dce925-a06c-416e-82fd-34c7a48147ec/image16.png

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6a398ed1-08df-4f21-b0aa-ced5698a23f4/image17.png

실내 주차장 측정

(1803040, 1) (1180, 1528)

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e1211cc7-a44f-49ba-a31f-8c9e80c783fb/image18.png

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/10e13d0c-df46-486f-9fb1-6e8abfbdeeef/image19.png

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dd401c2a-208a-44de-aa58-9dc1243f1517/image20.png

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9c08f89f-a3c8-4dc9-a293-b9bf396a0024/image21.png

사람

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/411dd823-52b5-4773-9f09-3d725cdbe75d/image22.png

일레클

!https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9e026dbe-eae7-43a1-89ee-d3bcb1ad6dbb/image23.png
