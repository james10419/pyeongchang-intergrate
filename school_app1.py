import pandas as pd
import math
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import sys 
from io import StringIO # StringIO 추가

# Haversine 공식을 사용하여 두 지점 간의 거리를 계산하는 함수 (킬로미터 단위)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    try:
        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2_rad = math.radians(float(lat2))
        lon2_rad = math.radians(float(lon2))
    except ValueError:
        raise ValueError("위도 또는 경도 값이 숫자가 아닙니다.")

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def determine_school_type_from_name(school_name_str):
    """
    학교 이름 문자열을 기반으로 학교급을 단순 추론합니다.
    주의: 이 방식은 학교 이름이 일정한 패턴('초', '중', '고' 등 포함)을 따를 때만 유효하며,
    모든 경우를 정확히 판별하기 어려울 수 있습니다.
    실제 적용 시 더 정교한 학교급 판별 방법(예: 별도 데이터 참조)을 고려해야 합니다.
    """
    name_cleaned = school_name_str.strip()
    if "초등학교" in name_cleaned or (name_cleaned.endswith("초") and not name_cleaned.endswith("중초") and not name_cleaned.endswith("고초")):
        return 'elementary'
    elif "중학교" in name_cleaned or name_cleaned.endswith("중") or \
         "고등학교" in name_cleaned or name_cleaned.endswith("고"):
        return 'secondary'
    else:
        return 'unknown' # 알 수 없는 학교급

def propose_school_consolidation(df, config, output_widget):
    """
    학교 통폐합 제안 로직을 수행하고 결과를 GUI 위젯에 출력하며, 상세 결과 리스트를 반환합니다.
    :param df: 학교 데이터가 포함된 pandas DataFrame
    :param config: 통폐합 기준 설정값 (딕셔너리)
    :param output_widget: 전체 분석 결과를 출력할 tkinter ScrolledText 위젯
    :return: (consolidation_decisions, unconsolidated_schools_info, final_active_schools, closed_schools_summary) 또는 오류 시 (None, None, None, None)
    """
    output_widget.insert(tk.END, "분석 시작...\n")
    output_widget.see(tk.END) 
    
    col_map = config['excel_column_names'] 
    required_cols_internal_keys = ['name', 'baseline_students', 'latitude', 'longitude'] 
    required_actual_col_names = [col_map[key] for key in required_cols_internal_keys if key in col_map] 
    
    missing_cols = [actual_name for actual_name in required_actual_col_names if actual_name not in df.columns]
    
    if missing_cols:
        error_msg = f"오류: 입력된 데이터에 다음 필수 컬럼명이 존재하지 않습니다: {', '.join(missing_cols)}\n\n"
        error_msg += "GUI 화면의 '데이터 컬럼명 설정' 부분에 입력된 각 항목의 이름이\n"
        error_msg += "실제 엑셀 파일 또는 붙여넣은 데이터의 첫 번째 행(헤더)에 있는\n"
        error_msg += "컬럼명과 정확히 일치하는지 확인해주십시오.\n\n"
        error_msg += f"현재 프로그램이 찾고 있는 필수 컬럼명 (사용자 설정 기준):\n"
        for key in required_cols_internal_keys: 
            if key in col_map:
                 error_msg += f"  - 내부 '{key}' 항목에 대해 설정된 컬럼명: '{col_map[key]}'\n"
        error_msg += f"\n입력 데이터에 실제 존재하는 컬럼명: {', '.join(df.columns.tolist())}"
        messagebox.showerror("데이터 컬럼 오류", error_msg)
        output_widget.insert(tk.END, error_msg + "\n")
        return None, None, None, None

    active_schools = [] 
    school_id_counter = 0 
    warnings = [] 

    annual_student_cols_str = col_map.get('year_student_counts_str', '')
    annual_student_cols = [col.strip() for col in annual_student_cols_str.split(',') if col.strip()] if annual_student_cols_str else []

    for index, row in df.iterrows():
        try:
            school_name = str(row[col_map['name']])
            school_type = determine_school_type_from_name(school_name)

            if school_type == 'unknown':
                warnings.append(f"경고: '{school_name}' 학교의 학교급을 이름에서 추론할 수 없어 분석에서 제외합니다. (이름에 '초', '중', '고' 또는 '초등학교', '중학교', '고등학교' 등이 포함되어야 함)")
                continue
            
            try:
                students = int(row[col_map['baseline_students']]) 
                latitude = float(row[col_map['latitude']])
                longitude = float(row[col_map['longitude']])
            except ValueError as ve:
                warnings.append(f"경고: '{school_name}' 학교의 데이터 형식에 오류가 있어 제외합니다 (기준 학생 수, 위도, 경도 중 하나가 숫자가 아님: {ve}). 해당 행 데이터: {row.to_dict()}")
                continue
            except KeyError as ke:
                 warnings.append(f"경고: '{school_name}' 학교 처리 중 설정된 컬럼명 '{str(ke)}'을 데이터에서 찾을 수 없습니다. '데이터 컬럼명 설정'을 확인하세요.")
                 continue
            
            annual_data = {}
            for col_name in annual_student_cols:
                if col_name in row:
                    try:
                        annual_data[col_name] = int(row[col_name])
                    except ValueError:
                        warnings.append(f"경고: '{school_name}' 학교의 연도별 학생 수 컬럼 '{col_name}'의 값('{row[col_name]}')이 숫자가 아니어서 해당 연도 값은 제외합니다.")
                        annual_data[col_name] = None 
                else:
                    warnings.append(f"경고: '{school_name}' 학교 데이터에 연도별 학생 수 컬럼 '{col_name}'이(가) 없어 해당 연도 데이터를 읽지 못했습니다.")

            active_schools.append({
                'id': school_id_counter,
                'name': school_name,
                'original_students': students, 
                'current_students': students,  
                'latitude': latitude,
                'longitude': longitude,
                'type': school_type,          
                'status': 'active',
                'annual_student_counts': annual_data 
            })
            school_id_counter += 1
        except KeyError as ke: 
            warnings.append(f"경고: 데이터 처리 중 내부 설정 오류({ke}). '데이터 컬럼명 설정'과 실제 데이터 헤더를 확인하세요. 오류 발생 행 인덱스: {index}")
            continue
        except Exception as e: 
            school_name_for_error = "알 수 없는 학교 (이름 컬럼 접근 불가)"
            try:
                school_name_for_error = str(row[col_map['name']])
            except:
                pass 
            warnings.append(f"경고: '{school_name_for_error}' 학교 데이터 처리 중 예기치 않은 오류 발생({e}). 해당 행 인덱스: {index}")
            continue
    
    if warnings:
        output_widget.insert(tk.END, "\n--- 데이터 처리 중 경고 ---\n")
        for warn in warnings:
            output_widget.insert(tk.END, warn + "\n")
        output_widget.insert(tk.END, "--------------------------\n\n")
        output_widget.see(tk.END)

    if not active_schools:
        messagebox.showinfo("정보", "분석 가능한 학교 데이터가 없습니다. 입력된 데이터나 설정을 다시 확인해주세요.")
        output_widget.insert(tk.END, "분석 가능한 학교 데이터가 없습니다.\n")
        return None, None, None, None

    consolidation_decisions = [] 
    unconsolidated_schools_info = [] 

    iteration_count = 0 
    max_iterations = len(active_schools) * 2 

    while iteration_count < max_iterations:
        iteration_count += 1
        schools_below_threshold = [] 
        for school in active_schools:
            if school['status'] != 'active': 
                continue
            
            threshold = config['student_threshold_elementary'] if school['type'] == 'elementary' \
                        else config['student_threshold_secondary']
            
            if school['current_students'] < threshold: 
                schools_below_threshold.append(school)

        if not schools_below_threshold:
            break

        schools_below_threshold.sort(key=lambda s: s['current_students'])
        closing_school = schools_below_threshold[0] 

        potential_absorbers = [] 
        radius_km = config['radius_elementary_km'] if closing_school['type'] == 'elementary' \
                    else config['radius_secondary_km']

        for candidate_school in active_schools:
            if candidate_school['id'] == closing_school['id'] or \
               candidate_school['type'] != closing_school['type'] or \
               candidate_school['status'] != 'active': 
                continue
            
            try:
                distance = haversine(closing_school['latitude'], closing_school['longitude'],
                                     candidate_school['latitude'], candidate_school['longitude'])
            except ValueError as e: 
                output_widget.insert(tk.END, f"거리 계산 오류: '{closing_school['name']}' 또는 '{candidate_school['name']}'의 좌표값 확인 필요. ({e})\n")
                continue 

            if distance <= radius_km: 
                potential_absorbers.append({
                    'school_data': candidate_school,
                    'distance': distance
                })
        
        if not potential_absorbers:
            closing_school['status'] = 'unconsolidated_no_partner'
            unconsolidated_schools_info.append({
                'name': closing_school['name'],
                'students': closing_school['current_students'], 
                'type': closing_school['type'],
                'reason': f'{radius_km}km 반경 내 통폐합 가능한 동일 학교급 학교 없음',
                'annual_student_counts': closing_school.get('annual_student_counts', {})
            })
        else:
            potential_absorbers.sort(key=lambda p: p['distance'])
            chosen_absorber_info = potential_absorbers[0]
            absorbing_school_dict = chosen_absorber_info['school_data'] 
            distance_to_absorber = chosen_absorber_info['distance']

            absorbing_school_original_students = 0 
            absorbing_school_new_students = 0 

            for sch in active_schools:
                if sch['id'] == absorbing_school_dict['id']:
                    absorbing_school_original_students = sch['current_students'] 
                    sch['current_students'] += closing_school['current_students'] 
                    absorbing_school_new_students = sch['current_students'] 
                    break
            
            closing_school['status'] = 'closed' 
            max_suggested_commute_km = distance_to_absorber + radius_km

            consolidation_decisions.append({
                'closed_school_name': closing_school['name'],
                'closed_school_students': closing_school['current_students'], 
                'absorbing_school_name': absorbing_school_dict['name'],
                'absorbing_school_original_students_before_merge': absorbing_school_original_students,
                'absorbing_school_new_students_after_merge': absorbing_school_new_students,
                'distance_km': round(distance_to_absorber, 2),
                'max_suggested_commute_km': round(max_suggested_commute_km, 2),
                'school_type': closing_school['type'],
                'closed_school_annual_counts': closing_school.get('annual_student_counts', {})
            })

    final_active_schools = [s for s in active_schools if s['status'] == 'active'] 
    closed_schools_summary = [ 
        {
            'name': s['name'], 
            'original_students': s['original_students'], 
            'type': s['type'],
            'status': '폐교됨 (통폐합)',
            'annual_student_counts': s.get('annual_student_counts', {})
        } for s in active_schools if s['status'] == 'closed'
    ]

    output_widget.insert(tk.END, "\n--- 통폐합 결정 사항 ---\n")
    if consolidation_decisions: 
        for dec in consolidation_decisions:
            output_widget.insert(tk.END, f"폐교 대상: {dec['closed_school_name']} (기준 학생수: {dec['closed_school_students']}명, 학교급: {dec['school_type']})\n")
            output_widget.insert(tk.END, f"  -> 통합 학교: {dec['absorbing_school_name']} (통합 전 기준 학생수: {dec['absorbing_school_original_students_before_merge']}명 -> 통합 후 기준 학생수: {dec['absorbing_school_new_students_after_merge']}명)\n")
            output_widget.insert(tk.END, f"  거리: {dec['distance_km']}km, 최대 제안 통학 거리: {dec['max_suggested_commute_km']}km\n")
            output_widget.insert(tk.END, "-" * 30 + "\n")
    else:
        output_widget.insert(tk.END, "통폐합 결정 사항이 없습니다.\n")

    output_widget.insert(tk.END, "\n--- 통폐합 불가 학교 ---\n")
    if unconsolidated_schools_info: 
        for school in unconsolidated_schools_info:
            output_widget.insert(tk.END, f"학교명: {school['name']} (기준 학생수: {school['students']}명, 학교급: {school['type']}) - 사유: {school['reason']}\n")
    else:
        output_widget.insert(tk.END, "통폐합 불가 학교가 없습니다.\n")

    output_widget.insert(tk.END, "\n--- 최종 운영 학교 현황 ---\n")
    if final_active_schools: 
        for school in final_active_schools:
            output_widget.insert(tk.END, f"학교명: {school['name']}, 현재 기준 학생 수: {school['current_students']}, 학교급: {school['type']}, (초기 기준 학생 수: {school['original_students']})\n")
    else:
        output_widget.insert(tk.END, "최종 운영 학교가 없습니다.\n")
        
    output_widget.insert(tk.END, "\n--- 폐교된 학교 요약 ---\n")
    if closed_schools_summary: 
        for school in closed_schools_summary:
            output_widget.insert(tk.END, f"학교명: {school['name']}, (초기 기준 학생 수: {school['original_students']}), 학교급: {school['type']}, 상태: {school['status']}\n")
    else:
        output_widget.insert(tk.END, "폐교된 학교가 없습니다.\n")
    
    output_widget.insert(tk.END, "\n분석 완료.\n")
    output_widget.see(tk.END)
    
    return consolidation_decisions, unconsolidated_schools_info, final_active_schools, closed_schools_summary


class SchoolConsolidationApp:
    def __init__(self, master):
        self.master = master
        master.title("학교 통폐합 제언 프로토타입")
        master.geometry("800x850") 

        self.analysis_results = None 

        style = ttk.Style()
        style.theme_use('clam') 
        style.configure("TLabel", padding=5, font=('Helvetica', 10))
        style.configure("TButton", padding=5, font=('Helvetica', 10))
        style.configure("TEntry", padding=5, font=('Helvetica', 10))
        style.configure("TFrame", padding=10)
        style.configure("Accent.TButton", foreground="white", background="#007bff")


        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.notebook = ttk.Notebook(main_frame)
        
        self.file_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.file_tab, text='파일 업로드')

        file_input_frame = ttk.LabelFrame(self.file_tab, text="엑셀 파일 선택", padding=10)
        file_input_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_label = ttk.Label(file_input_frame, text="선택된 파일 없음") 
        self.file_path_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,10))
        self.select_file_button = ttk.Button(file_input_frame, text="파일 찾기...", command=self.select_file) 
        self.select_file_button.pack(side=tk.RIGHT)
        self.excel_file_path = None 

        self.paste_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.paste_tab, text='데이터 직접 입력')

        paste_instruct_label = ttk.Label(self.paste_tab, 
                                         text="엑셀 데이터를 복사하여 아래 텍스트 영역에 붙여넣으세요.\n(주의: 첫 번째 행은 반드시 컬럼 이름(헤더)이어야 하며, 각 데이터는 탭(Tab)으로 구분되어야 합니다.)\n'데이터 컬럼명 설정'과 헤더명이 일치해야 합니다.", 
                                         justify=tk.LEFT)
        paste_instruct_label.pack(pady=(0,5), anchor='w')
        
        self.pasted_data_area = scrolledtext.ScrolledText(self.paste_tab, wrap=tk.NONE, width=70, height=10, font=('Courier New', 9)) 
        self.pasted_data_area.pack(fill=tk.BOTH, expand=True)
        
        self.notebook.pack(expand=True, fill='both', pady=5)

        config_outer_frame = ttk.Frame(main_frame) 
        config_outer_frame.pack(fill=tk.X, pady=5)

        config_frame = ttk.LabelFrame(config_outer_frame, text="통폐합 기준 설정", padding=10)
        config_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))

        ttk.Label(config_frame, text="초등학생 수 기준 (이하):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.threshold_elem_entry = ttk.Entry(config_frame, width=10)
        self.threshold_elem_entry.grid(row=0, column=1, sticky=tk.W, pady=2)
        self.threshold_elem_entry.insert(0, "60") 

        ttk.Label(config_frame, text="중고등학생 수 기준 (이하):").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(10,0))
        self.threshold_sec_entry = ttk.Entry(config_frame, width=10)
        self.threshold_sec_entry.grid(row=0, column=3, sticky=tk.W, pady=2)
        self.threshold_sec_entry.insert(0, "100") 

        ttk.Label(config_frame, text="초등학교 통폐합 반경(km):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.radius_elem_entry = ttk.Entry(config_frame, width=10)
        self.radius_elem_entry.grid(row=1, column=1, sticky=tk.W, pady=2)
        self.radius_elem_entry.insert(0, "2.0") 

        ttk.Label(config_frame, text="중고등학교 통폐합 반경(km):").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(10,0))
        self.radius_sec_entry = ttk.Entry(config_frame, width=10)
        self.radius_sec_entry.grid(row=1, column=3, sticky=tk.W, pady=2)
        self.radius_sec_entry.insert(0, "15.0") 

        excel_col_frame = ttk.LabelFrame(config_outer_frame, text="데이터 컬럼명 설정", padding=10)
        excel_col_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5,0))
        
        self.col_config_map = {
            'name': {'label': '학교명 컬럼명:', 'default': '학교명'},
            'baseline_students': {'label': '기준 학생수 컬럼명:', 'default': '학생수'},
            'latitude': {'label': '위도 컬럼명:', 'default': '위도'},
            'longitude': {'label': '경도 컬럼명:', 'default': '경도'},
            'year_student_counts_str': {'label': '연도별 학생수 컬럼명(들)\n(쉼표로 구분, 선택 사항):', 'default': ''} 
        }
        
        self.col_entries = {} 
        row_idx = 0
        for key, config_data in self.col_config_map.items():
            ttk.Label(excel_col_frame, text=config_data['label'], justify=tk.LEFT).grid(row=row_idx, column=0, sticky=tk.W, pady=1, padx=2)
            entry = ttk.Entry(excel_col_frame, width=20) 
            entry.grid(row=row_idx, column=1, sticky=tk.W, pady=1, padx=2)
            entry.insert(0, config_data['default']) 
            self.col_entries[key] = entry 
            row_idx +=1
        
        self.run_button = ttk.Button(main_frame, text="분석 실행", command=self.run_analysis, style="Accent.TButton")
        self.run_button.pack(pady=10)

        result_frame = ttk.LabelFrame(main_frame, text="전체 분석 결과", padding=10) 
        result_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=80, height=15, font=('Helvetica', 10), state=tk.DISABLED) 
        self.result_text.pack(fill=tk.BOTH, expand=True)

        search_frame = ttk.LabelFrame(main_frame, text="개별 학교 검색", padding=10)
        search_frame.pack(pady=10, padx=0, fill=tk.X)

        search_input_frame = ttk.Frame(search_frame)
        search_input_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(search_input_frame, text="학교명 입력:").pack(side=tk.LEFT, padx=(0,5))
        self.search_school_entry = ttk.Entry(search_input_frame, width=40) 
        self.search_school_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        
        self.search_school_button = ttk.Button(search_input_frame, text="검색", command=self.search_individual_school_info) 
        self.search_school_button.pack(side=tk.LEFT)

        self.individual_search_result_text = scrolledtext.ScrolledText(search_frame, wrap=tk.WORD, width=70, height=6, font=('Helvetica', 10), state=tk.DISABLED) 
        self.individual_search_result_text.pack(fill=tk.BOTH, expand=True, pady=(5,0))


    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="분석할 엑셀 파일을 선택하세요",
            filetypes=(("Excel files", "*.xlsx *.xls"), ("All files", "*.*"))
        )
        if file_path:
            self.excel_file_path = file_path
            self.file_path_label.config(text=file_path.split('/')[-1]) 
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END) 
            self.result_text.insert(tk.END, f"선택된 파일: {file_path}\n'분석 실행' 버튼을 눌러주세요.\n")
            self.result_text.config(state=tk.DISABLED)
            self.analysis_results = None 
            self.individual_search_result_text.config(state=tk.NORMAL)
            self.individual_search_result_text.delete('1.0', tk.END)
            self.individual_search_result_text.config(state=tk.DISABLED)
        else:
            self.excel_file_path = None
            self.file_path_label.config(text="선택된 파일 없음")


    def run_analysis(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete('1.0', tk.END) 
        self.individual_search_result_text.config(state=tk.NORMAL)
        self.individual_search_result_text.delete('1.0', tk.END) 
        self.individual_search_result_text.config(state=tk.DISABLED)
        self.analysis_results = None 
        
        try:
            config_values = {
                'student_threshold_elementary': int(self.threshold_elem_entry.get()),
                'student_threshold_secondary': int(self.threshold_sec_entry.get()),
                'radius_elementary_km': float(self.radius_elem_entry.get()),
                'radius_secondary_km': float(self.radius_sec_entry.get()),
            }
            excel_col_names_config = {}
            for key, entry_widget in self.col_entries.items():
                excel_col_names_config[key] = entry_widget.get().strip()
            
            config_values['excel_column_names'] = excel_col_names_config

        except ValueError:
            messagebox.showerror("입력 오류", "통폐합 기준 설정값이 올바른 숫자 형식이 아닙니다. 확인 후 다시 시도해주세요.")
            self.result_text.config(state=tk.DISABLED)
            return

        df = None 
        selected_tab_index = self.notebook.index(self.notebook.select()) 

        if selected_tab_index == 0: 
            if not self.excel_file_path:
                messagebox.showerror("파일 오류", "먼저 엑셀 파일을 선택해주세요.")
                self.result_text.config(state=tk.DISABLED)
                return
            try:
                df = pd.read_excel(self.excel_file_path)
                self.result_text.insert(tk.END, f"파일에서 데이터 로드 완료: {self.excel_file_path}\n")
            except Exception as e:
                messagebox.showerror("파일 읽기 오류", f"엑셀 파일 읽기 중 오류 발생: {e}\n파일이 열려있거나, 형식이 올바르지 않거나, 필요한 라이브러리(openpyxl)가 설치되지 않았을 수 있습니다.")
                self.result_text.insert(tk.END, f"엑셀 파일 읽기 오류: {e}\n")
                self.result_text.config(state=tk.DISABLED)
                return
        elif selected_tab_index == 1: 
            pasted_content = self.pasted_data_area.get("1.0", tk.END).strip() 
            if not pasted_content:
                messagebox.showerror("데이터 오류", "붙여넣을 데이터를 입력해주세요.")
                self.result_text.config(state=tk.DISABLED)
                return
            try:
                df = pd.read_csv(StringIO(pasted_content), sep='\t', engine='python')
                self.result_text.insert(tk.END, "붙여넣은 데이터 로드 완료.\n")
            except Exception as e:
                messagebox.showerror("데이터 파싱 오류", f"붙여넣은 데이터 파싱 중 오류 발생: {e}\n첫 행은 헤더, 데이터는 탭(Tab)으로 구분되었는지, '데이터 컬럼명 설정'이 올바른지 확인해주세요.")
                self.result_text.insert(tk.END, f"데이터 파싱 오류: {e}\n")
                self.result_text.config(state=tk.DISABLED)
                return
        
        if df is not None:
            if df.empty:
                messagebox.showinfo("정보", "로드된 데이터가 비어있습니다. 입력 내용을 확인해주세요.")
                self.result_text.insert(tk.END, "로드된 데이터가 비어있습니다.\n")
                self.result_text.config(state=tk.DISABLED)
                return
            
            results = propose_school_consolidation(df, config_values, self.result_text) 
            if results and results[0] is not None: # 첫 번째 결과 요소(decisions)가 None이 아니면 정상으로 간주
                decisions, unconsolidated, final_schools, closed_summary = results
                self.analysis_results = {
                    'decisions': decisions,
                    'unconsolidated': unconsolidated,
                    'final_schools': final_schools,
                    'closed_summary': closed_summary 
                }
            else: 
                self.analysis_results = None 
                self.result_text.insert(tk.END, "분석 중 오류가 발생하여 상세 결과를 저장하지 못했습니다.\n")
        else:
            self.result_text.insert(tk.END, "데이터를 로드하지 못했습니다. 입력 방식과 내용을 다시 확인해주세요.\n")
        
        self.result_text.config(state=tk.DISABLED) 

    def search_individual_school_info(self):
        """개별 학교 정보를 검색하여 결과를 표시합니다."""
        if not self.analysis_results or \
           self.analysis_results.get('decisions') is None or \
           self.analysis_results.get('unconsolidated') is None or \
           self.analysis_results.get('final_schools') is None:
            messagebox.showinfo("정보", "먼저 전체 분석을 실행해주세요. 저장된 분석 결과가 없습니다.")
            return

        school_name_query = self.search_school_entry.get().strip()
        if not school_name_query:
            messagebox.showinfo("정보", "검색할 학교명을 입력해주세요.")
            return

        self.individual_search_result_text.config(state=tk.NORMAL) 
        self.individual_search_result_text.delete('1.0', tk.END) 

        found_info = False
        result_message = ""
        school_name_query_lower = school_name_query.lower() 

        # 1. 폐교 대상인지 확인
        for dec in self.analysis_results['decisions']:
            if dec['closed_school_name'].lower() == school_name_query_lower:
                annual_counts_str = ", ".join([f"{k}: {v}" for k, v in dec.get('closed_school_annual_counts', {}).items() if v is not None])
                result_message = (
                    f"학교명: {dec['closed_school_name']}\n" 
                    f"상태: 폐교 대상 (통폐합됨)\n"
                    f"사유: 학생 수 기준 미달 및 통폐합 조건 만족\n"
                    f"통합된 학교: {dec['absorbing_school_name']}\n"
                    f"기준 학생수: {dec['closed_school_students']}\n"
                    f"거리: {dec['distance_km']}km, 최대 제안 통학 거리: {dec['max_suggested_commute_km']}km\n"
                    f"연도별 학생수: {annual_counts_str if annual_counts_str else '정보 없음'}"
                )
                found_info = True
                break
        if found_info:
            self.individual_search_result_text.insert(tk.END, result_message)
            self.individual_search_result_text.config(state=tk.DISABLED) 
            return

        # 2. 통폐합 불가 학교인지 확인
        for school in self.analysis_results['unconsolidated']:
            if school['name'].lower() == school_name_query_lower:
                annual_counts_str = ", ".join([f"{k}: {v}" for k, v in school.get('annual_student_counts', {}).items() if v is not None])
                result_message = (
                    f"학교명: {school['name']}\n" 
                    f"상태: 통폐합 불가\n"
                    f"사유: {school['reason']}\n"
                    f"당시 기준 학생수: {school['students']}, 학교급: {school['type']}\n"
                    f"연도별 학생수: {annual_counts_str if annual_counts_str else '정보 없음'}"
                )
                found_info = True
                break
        if found_info:
            self.individual_search_result_text.insert(tk.END, result_message)
            self.individual_search_result_text.config(state=tk.DISABLED)
            return

        # 3. 최종 운영 학교인지 확인
        for school in self.analysis_results['final_schools']:
            if school['name'].lower() == school_name_query_lower:
                absorbed_schools_list = []
                for dec_item in self.analysis_results['decisions']:
                    if dec_item['absorbing_school_name'].lower() == school['name'].lower(): 
                        absorbed_schools_list.append(f"  - {dec_item['closed_school_name']} (기준 학생수: {dec_item['closed_school_students']}명)")
                
                annual_counts_str = ", ".join([f"{k}: {v}" for k, v in school.get('annual_student_counts', {}).items() if v is not None])

                if absorbed_schools_list:
                    absorbed_schools_str = "\n".join(absorbed_schools_list)
                    result_message = (
                        f"학교명: {school['name']}\n" 
                        f"상태: 최종 운영 학교 (통합 학교)\n"
                        f"사유: 기준 충족 및 타 학교 통합\n"
                        f"현재 기준 학생 수: {school['current_students']} (초기 기준 학생 수: {school['original_students']})\n"
                        f"연도별 학생수: {annual_counts_str if annual_counts_str else '정보 없음'}\n"
                        f"통합한 학교 목록:\n{absorbed_schools_str}"
                    )
                else:
                    result_message = (
                        f"학교명: {school['name']}\n" 
                        f"상태: 최종 운영 학교\n"
                        f"사유: 학생 수 기준 충족 또는 통폐합 대상 아님\n"
                        f"현재 기준 학생 수: {school['current_students']} (초기 기준 학생 수: {school['original_students']})\n"
                        f"연도별 학생수: {annual_counts_str if annual_counts_str else '정보 없음'}"
                    )
                found_info = True
                break
                
        if found_info:
            self.individual_search_result_text.insert(tk.END, result_message)
        else:
            self.individual_search_result_text.insert(tk.END, f"'{school_name_query}' 학교 정보를 찾을 수 없습니다.\n학교명을 정확히 입력했는지, 또는 전체 분석이 올바르게 완료되었는지 확인해주세요.")
        
        self.individual_search_result_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    try:
        root = tk.Tk() 
        app = SchoolConsolidationApp(root) 
        root.mainloop() 
    except tk.TclError as e:
        if "no display name" in str(e).lower() or "display" in str(e).lower():
            print("--- GUI 실행 오류 ---", file=sys.stderr)
            print("Tkinter GUI 애플리케이션을 실행하기 위한 디스플레이 환경을 찾을 수 없습니다.", file=sys.stderr)
            print("이 프로그램은 그래픽 사용자 인터페이스(GUI)를 사용하므로, 데스크톱 환경에서 실행해야 합니다.", file=sys.stderr)
            print("원인: $DISPLAY 환경 변수가 설정되지 않았거나, X 서버에 접속할 수 없는 환경일 수 있습니다.", file=sys.stderr)
            print("해결책:", file=sys.stderr)
            print("  1. 일반적인 데스크톱 환경(Windows, macOS, Linux 데스크톱)에서 프로그램을 실행하고 있는지 확인하세요.", file=sys.stderr)
            print("  2. SSH를 통해 원격 서버에 접속한 경우, X11 포워딩 옵션(예: ssh -X user@host)을 사용해야 할 수 있습니다.", file=sys.stderr)
            print("  3. Docker 컨테이너 내부에서 실행하는 경우, X 서버 연동을 위한 추가 설정이 필요합니다.", file=sys.stderr)
            print(f"\n상세 오류 메시지: {e}", file=sys.stderr)
        else:
            print(f"Tkinter 관련 예기치 않은 오류 발생: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1) 
    except Exception as e:
        print(f"애플리케이션 실행 중 예기치 않은 오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)