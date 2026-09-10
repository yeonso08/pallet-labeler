# Changelog

이 프로젝트의 주요 변경사항을 버전별로 기록합니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 따르고,
버전 번호는 [Semantic Versioning](https://semver.org/lang/ko/)을 따릅니다.

## [Unreleased]

### Added
- `theme.py`: 화면 색상·글꼴·컨트롤 스타일을 한곳에서 관리합니다.

### Changed
- 화면 전체를 점군 캔버스와 같은 어두운 색으로 통일했습니다. 강조색은 스캔 속 팔레트 목재의 주황색이며, 저장 버튼과 켜져 있는 표시 버튼에만 사용합니다. **위젯 배치는 그대로입니다.**
- 팔레트·번호 표시가 체크박스에서 켜짐/꺼짐 버튼으로 바뀌었습니다. clam 테마의 체크 표시가 ✕ 모양이라 "해제"처럼 읽혔습니다.
- 3D 회전·확대와 2D 이동이 훨씬 부드러워졌습니다. 화면이 움직이는 동안에는 점을 2만 개만 굵게 그리고, 멈추면 전체 밀도로 다시 그립니다 (58만 점 기준 프레임당 약 404ms → 50ms).
- 색상·밀도·표시 변경과 2D/3D 전환이 성긴 화면을 먼저 그린 뒤 채우는 방식으로 바뀌어 즉시 반응합니다 (약 442ms → 78ms).

### Fixed
- 물체 번호 위치 계산과 표시 점 추출을 캐시해, 다시 그릴 때마다 전체 점을 반복 스캔하지 않습니다.

## [2.1.0] - 2026-09-10

레포 초기 커밋 시점 기준. `사용법.md`에 기록된 "v2.1 화면 개선" 이후 상태를 반영합니다.
이전 v1~v2 개발 이력은 `검증기록.md`, `추가학습_결과.md`를 참고하세요.

### Added
- 팔레트/부자재 라벨링 앱 (`app.py`, `engine.py`, `viewer.py`)
- ExtraTrees 기반 분류 모델 및 학습/재학습 스크립트 (`train.py`, `retrain.py`)
- v1/v2 모델 파일 및 평가 결과 (`model.joblib`, `models/model_v1.joblib`, `evaluation.json`, `evaluation_v2.json`)
- 사용법 및 검증 기록 문서

[Unreleased]: https://github.com/yeonso08/pallet-labeler/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/yeonso08/pallet-labeler/releases/tag/v2.1.0
