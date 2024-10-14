import pandas as pd
import os
import itertools
import re

# 1. 지점별 입력 값 (location_1, location_2, location_3에 해당하는 값들)
# location_1 = (lat_1, lon_1, dist_1, sbd_1)
# location_2 = (lat_2, lon_2, dist_2, sbd_2)
# location_3 = (lat_3, lon_3, dist_3, sbd_3)
# depths = (depth_1, depth_2, depth_3)

location_1 = (37.7050950, 128.0318270, 26, 1.44)  # E1    '19850': location_1,
location_2 = (37.7052430, 128.0322080, 62, 1.44)  # E2    '19846': location_2,
location_3 = (37.7055270, 128.0324520, 99, 1.44)  # E3    '19853': location_3,
location_4 = (37.7057970, 128.0326820, 135, 1.44)  # E4   '19843': location_4,
location_5 = (37.7057230, 128.0311750, 99, 1.44)  # N1    '05589': location_5,
location_6 = (37.7058490, 128.0309820, 119, 1.44)  # N2   '19848': location_6,
location_7 = (37.7049540, 128.0311020, 48, 1.44)  # W1    '19852': location_7,
location_8 = (37.7048480, 128.0309160, 65, 1.44)  # W2    '19851': location_8,
location_9 = (37.7046330, 128.0313980, 37, 1.44)  # S1    '19854': location_9,
location_10 = (37.7042950, 128.0316150, 68, 1.44)  # S2   '19847': location_10,
location_11 = (37.7040370, 128.0319370, 100, 1.44)  # S3  '19903': location_11,
location_12 = (37.7038900, 128.0316900, 113, 1.44)  # S4  '19845': location_12
depths = (10, 20, 30)  # 각 센서의 깊이

# 폴더 경로 및 파일 저장 경로
input_folder = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\01.Input\01.FDR"
output_file = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\HC_FDR_input.xlsx"

# 파일 이름에 따라 location 매칭
file_location_mapping = {
    '19850': location_1,
    '19846': location_2,
    '19853': location_3,
    '19843': location_4,
    '05589': location_5,
    '19848': location_6,
    '19852': location_7,
    '19851': location_8,
    '19854': location_9,
    '19847': location_10,
    '19903': location_11,
    '19845': location_12
}

all_data = []  # 결과를 저장할 리스트

# 폴더 내 엑셀 파일들을 하나로 통합
for file in os.listdir(input_folder):
    if file.endswith('.xlsx') or file.endswith('.csv'):
        file_path = os.path.join(input_folder, file)

        # 파일 이름에서 기호, 괄호 등을 제거하고 소문자로 변환
        simplified_file_name = re.sub(r'[^\w\s]', '', file).lower()

        # 파일 읽기 (엑셀 또는 CSV에 따라 처리)
        if file.endswith('.xlsx'):
            df = pd.read_excel(file_path, skiprows=2)  # 4행부터 데이터가 있으므로 2행 스킵
        elif file.endswith('.csv'):
            df = pd.read_csv(file_path, skiprows=2)  # CSV 파일인 경우도 2행 스킵
        
        # 필요한 열 선택 및 이름 지정
        df_selected = df[['Timestamps', ' m3/m3 Water Content', ' m3/m3 Water Content.1', ' m3/m3 Water Content.2']].copy()  
        df_selected.columns = ['Date', 'theta_v_d1', 'theta_v_d2', 'theta_v_d3']
        
        # Date 열을 datetime 형식으로 변환
        df_selected['Date'] = pd.to_datetime(df_selected['Date'], errors='coerce')
        
        # 정각에만 해당하는 데이터 필터링 (분이 00인 데이터만 선택)
        df_selected = df_selected[df_selected['Date'].dt.minute == 0]
        
        # 파일 이름에서 location 매칭
        location_found = False
        for location_key, location_info in file_location_mapping.items():
            if location_key in simplified_file_name:  # 파일 이름이 location과 일치하는지 확인
                lat, lon, dist, bulk_density = location_info
                location_found = True
                break  # 해당 location이 매칭되면 루프를 종료
        
        # location 매칭 실패 시 오류 발생
        if not location_found:
            raise ValueError(f"File {file} does not match any known location keys.")
        
        # theta_v_d1, theta_v_d2, theta_v_d3 데이터를 하나로 합치기 (long format)
        df_long = pd.melt(df_selected, id_vars=['Date'],
                          value_vars=['theta_v_d1', 'theta_v_d2', 'theta_v_d3'],
                          var_name='theta_v_source', value_name='theta_v')

        # 같은 로거에서 읽어온 토양수분 자료들을 각각 depths 순서에 맞게 할당하기(i.e. 홍천의 경우 theta_d_1은 10, theta_d_2는 20, theta_d_3은 30)
        df_long['FDR_depth'] = df_long['theta_v_source'].map({
            'theta_v_d1': depths[0],
            'theta_v_d2': depths[1],
            'theta_v_d3': depths[2]
        })

        # 지점별 정보 추가
        df_long['latitude'] = lat
        df_long['longitude'] = lon
        df_long['distance_from_station'] = dist
        df_long['bulk_density'] = bulk_density

        # 최종 데이터 리스트에 추가
        all_data.append(df_long)

# 모든 데이터를 하나의 데이터프레임으로 병합
result_df = pd.concat(all_data, ignore_index=True)

# 결과를 Excel 파일로 저장
result_df.to_excel(output_file, index=False)

print(f"Data has been successfully saved to {output_file}")