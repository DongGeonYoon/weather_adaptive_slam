# 악천후 시뮬레이션 벤치마크

합성 도심 장면에서 LiDAR(32채널, 10 Hz)와 IMU(200 Hz) 데이터를 생성하고, 비·안개·눈·복합 날씨 모델을 적용해 FAST-LIO2의 여러 설정을 같은 bag으로 비교합니다. 실제 데이터셋은 필요 없습니다.

프로젝트 설명, 실험 결과와 고찰은 [저장소 README](../../README.md)에 있습니다.

## 파일

| 파일 | 역할 |
|---|---|
| `sim_world.py` | 장면(건물, 기둥, 나무, 지면)과 궤적을 정의합니다. 32채널 회전 LiDAR(모션 왜곡, `ring`/`time` 필드)와 IMU(노이즈, 바이어스)를 생성합니다. |
| `weather_models.py` | 날씨 모델입니다. 안개(Beer–Lambert 감쇠와 후방산란), 비(빗방울 반사와 젖은 노면), 눈(눈송이 클러터), 복합이 있고, 강도는 시간에 따라 변합니다. |
| `generate_dataset.py` | 날씨별 rosbag2(`/velodyne_points`, `/imu/data`), GT(TUM 형식), 원본 스캔을 생성합니다. |
| `run_experiments.sh` | 방법별로 `fastlio_mapping`을 실행하고 `/Odometry`와 `/cloud_weather_filtered`를 기록합니다. |
| `evaluate_sim.py` | ATE, RPE, 종단 드리프트, 맵 노이즈 비율, voxel IoU, 처리 시간을 계산합니다. |
| `compare_results.py` | 여러 데이터셋과 반복 실행의 결과를 합쳐 비교표와 기하평균 비율을 만듭니다. |
| `render_presentation.py` | 발표용 비교 영상(1920×1080, 한글), 지표 카드, 요약 차트, 몽타주를 만듭니다. |
| `make_highlight.sh` | 약 1분 40초짜리 하이라이트판을 만듭니다(주행 3배속, 지표 카드는 원래 속도). |

## 방법 (`run_experiments.sh`의 `METHODS`)

| 이름 | 설정 파일 | 설명 |
|---|---|---|
| `baseline` | `sim_weather_baseline.yaml` | FAST-LIO2 기본 |
| `original` | `sim_weather_weather_adaptive_original.yaml` | 기존 weather-adaptive |
| `proposed` | `sim_weather_weather_adaptive.yaml` | 최종 제안 방법 |
| `proposed_cov` | 위 파일 + `adaptive_lidar_cov.enable:=true` | 적응 공분산 ablation |

## 사용법

```bash
source /opt/ros/humble/setup.bash && source ~/ws_livox/install/setup.bash
source ../install/setup.bash
cd sim

# 1) 데이터셋 생성 (5종 날씨, 약 10분, 4~6 GB)
python3 generate_dataset.py --out ../sim_results/valC/data --world-seed 33 --speed 5.0 --seed-offset 3

# 2) 실행: 실시간 재생이라 bag 길이 × 방법 수 × 날씨 수만큼 걸립니다.
#    여러 데이터셋은 DOMAIN을 다르게 주면 동시에 돌릴 수 있습니다.
RES=$PWD/../sim_results/valC METHODS="baseline original proposed" DOMAIN=81 ./run_experiments.sh

# 3) 평가: <RES>/evaluation.json, <RES>/evaluation_table.txt
python3 evaluate_sim.py --results ../sim_results/valC --modes baseline original proposed

# 4) 데이터셋·반복 합산
python3 compare_results.py --dataset C=../sim_results/valC --dataset D=../sim_results/valD \
    --out ../sim_results/validation_comparison.txt

# 5) 발표 영상
python3 render_presentation.py --results ../sim_results/valC ../sim_results/valD \
    --labels "검증 데이터셋 C" "검증 데이터셋 D" --out ../sim_results/presentation
./make_highlight.sh ../sim_results/presentation     # 클립 구성은 스크립트의 CLIPS 배열에서 바꿉니다
```

- **반복 실행**: 같은 데이터로 반복하려면 새 결과 폴더를 만들고 `data`를 심볼릭 링크로 연결합니다. 평가와 렌더링은 `<RES>/data`를 읽습니다.
  ```bash
  mkdir -p ../sim_results/valC_rep2
  ln -s ../valC/data ../sim_results/valC_rep2/data
  ```
- **QoS 설정**: LiDAR 메시지는 하나에 약 1.2 MB입니다. 구독을 best-effort로 두면 DDS가 스캔의 상당수를 버리기 때문에, 실행 스크립트는 reliable QoS를 사용합니다.
- **비결정성**: FAST-LIO는 멀티스레드 타이밍 때문에 같은 입력에서도 ATE가 수 cm 정도 달라질 수 있습니다.
