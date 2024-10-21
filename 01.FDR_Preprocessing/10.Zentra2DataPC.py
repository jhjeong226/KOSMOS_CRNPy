import pandas as pd
import os
import re

# 폴더 경로 및 파일 저장 경로
input_folder = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\02.PC\01.Input\01.Zentra"
output_file = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\02.PC\02.Output\01.Preprocessed\PC_FDR_input.xlsx"
geoinfo_file = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\02.PC\01.Input\geo_locations.xlsx"

depths = (10, 30, 60)

geo_df = pd.read_excel(geoinfo_file, dtype={'loc_key':str})

geo_df['loc_key'] = geo_df['loc_key'].astype(str)
file_location_mapping = {
    str(row['loc_key']): (row['lat'], row['lon'], row['dist'], row['sbd'])
    for idx, row in geo_df.iterrows()
}

all_data = []  # 결과를 저장할 리스트

# 폴더 내 엑셀 파일들을 하나로 통합
for file in os.listdir(input_folder):
    if file.endswith('.xlsx') or file.endswith('.csv'):
        file_path = os.path.join(input_folder, file)

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
        
        # 파일 이름에서 loc_key 추출 (파일 이름에 19850 같은 숫자가 있으므로 해당 부분 추출)
        loc_key_match = re.search(r'\((z6-)?(\d+)\)', file)  # z6- 앞에 올 수 있는 형식 처리
        if loc_key_match:
            loc_key = loc_key_match.group(2)  # 그룹 2는 숫자만 추출
        else:
            raise ValueError(f"File {file} does not match any known loc_key.")

        # loc_key가 문자열로 변환되었는지 확인 후 매칭
        if loc_key not in file_location_mapping:
            raise ValueError(f"File {file} has loc_key {loc_key}, which does not match any known loc_key.")

        # location 매칭
        lat, lon, dist, bulk_density = file_location_mapping[loc_key]

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