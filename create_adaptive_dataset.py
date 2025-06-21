"""
为AdaptiveFS创建符合格式要求的医学问卷数据集

根据AdaptiveFS的要求创建：
1. small_data50.npy - 形状为(n,50)的实值numpy数组
2. names_small50.npy - 包含50个问题名称的numpy数组  
3. labels.npy - 对应的标签数组

数据集模拟真实的医学问卷调查，包含50个医学相关问题
"""

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def create_medical_question_names():
    """创建50个医学问卷问题名称"""
    
    question_names = [
        # 基本人口统计学信息 (0-9)
        'age_p',           # 年龄
        'sex',             # 性别 (1=男, 2=女)
        'hiscodi32',       # 种族编码 (0=非西班牙裔, 1=西班牙裔)
        'bmi',             # 体质量指数
        'education_level', # 教育水平
        'income_level',    # 收入水平
        'marital_status',  # 婚姻状况
        'employment_status', # 就业状况
        'insurance_status',  # 保险状况
        'region_code',     # 地区代码
        
        # 生活方式因素 (10-19)
        'smoking_status',      # 吸烟状况
        'alcohol_consumption', # 饮酒情况
        'exercise_frequency',  # 运动频率
        'diet_quality',        # 饮食质量
        'sleep_hours',         # 睡眠时间
        'stress_level',        # 压力水平
        'caffeine_intake',     # 咖啡因摄入
        'physical_activity',   # 体力活动
        'sedentary_time',      # 久坐时间
        'social_support',      # 社会支持
        
        # 症状和主诉 (20-29)
        'chest_pain',          # 胸痛
        'shortness_of_breath', # 呼吸困难
        'fatigue',             # 疲劳
        'dizziness',           # 头晕
        'headache',            # 头痛
        'nausea',              # 恶心
        'abdominal_pain',      # 腹痛
        'joint_pain',          # 关节痛
        'back_pain',           # 背痛
        'sleep_problems',      # 睡眠问题
        
        # 既往病史 (30-39)
        'diabetes_history',       # 糖尿病史
        'hypertension_history',   # 高血压史
        'heart_disease_history',  # 心脏病史
        'stroke_history',         # 中风史
        'cancer_history',         # 癌症史
        'kidney_disease_history', # 肾病史
        'liver_disease_history',  # 肝病史
        'lung_disease_history',   # 肺病史
        'mental_health_history',  # 精神健康史
        'family_heart_disease',   # 家族心脏病史
        
        # 实验室检查和生理指标 (40-49)
        'blood_pressure_systolic',  # 收缩压
        'blood_pressure_diastolic', # 舒张压
        'heart_rate',               # 心率
        'blood_glucose',            # 血糖
        'cholesterol_total',        # 总胆固醇
        'cholesterol_hdl',          # 高密度脂蛋白胆固醇
        'cholesterol_ldl',          # 低密度脂蛋白胆固醇
        'triglycerides',            # 甘油三酯
        'hemoglobin',               # 血红蛋白
        'white_blood_cell_count'    # 白细胞计数
    ]
    
    assert len(question_names) == 50, f"应该有50个问题，实际有{len(question_names)}个"
    
    return np.array(question_names, dtype='<U32')  # 使用Unicode字符串

def generate_realistic_medical_data(n_patients=1000, random_seed=42):
    """生成真实的医学问卷数据"""
    
    np.random.seed(random_seed)
    
    # 初始化数据矩阵
    data = np.zeros((n_patients, 50))
    
    # 生成基本人口统计学信息 (0-9)
    data[:, 0] = np.random.normal(45, 15, n_patients)  # age_p: 年龄，均值45，标准差15
    data[:, 0] = np.clip(data[:, 0], 18, 85)  # 限制在18-85岁
    
    data[:, 1] = np.random.choice([1, 2], n_patients, p=[0.48, 0.52])  # sex: 性别
    data[:, 2] = np.random.choice([0, 1], n_patients, p=[0.83, 0.17])  # hiscodi32: 种族
    
    # BMI: 正态分布，均值25，标准差5
    data[:, 3] = np.random.normal(25, 5, n_patients)
    data[:, 3] = np.clip(data[:, 3], 15, 50)
    
    # 其他人口统计学特征 (4-9)
    for i in range(4, 10):
        data[:, i] = np.random.normal(0, 1, n_patients)
    
    # 生活方式因素 (10-19)
    # 吸烟状况：0=从不，1=以前，2=现在
    data[:, 10] = np.random.choice([0, 1, 2], n_patients, p=[0.6, 0.25, 0.15])
    
    # 饮酒：0=从不，1=偶尔，2=经常
    data[:, 11] = np.random.choice([0, 1, 2], n_patients, p=[0.3, 0.5, 0.2])
    
    # 运动频率：天/周
    data[:, 12] = np.random.poisson(2.5, n_patients)
    data[:, 12] = np.clip(data[:, 12], 0, 7)
    
    # 其他生活方式因素 (13-19)
    for i in range(13, 20):
        data[:, i] = np.random.normal(0, 1, n_patients)
    
    # 症状和主诉 (20-29) - 大多数为0-3的严重程度评分
    for i in range(20, 30):
        # 使用泊松分布生成症状严重程度
        data[:, i] = np.random.poisson(0.5, n_patients)
        data[:, i] = np.clip(data[:, i], 0, 3)
    
    # 既往病史 (30-39) - 二元变量 0=无，1=有
    disease_prevalence = [0.08, 0.25, 0.06, 0.03, 0.04, 0.03, 0.02, 0.05, 0.15, 0.12]
    for i, prevalence in enumerate(disease_prevalence):
        data[:, 30+i] = np.random.choice([0, 1], n_patients, p=[1-prevalence, prevalence])
    
    # 实验室检查和生理指标 (40-49)
    # 收缩压
    data[:, 40] = np.random.normal(120, 15, n_patients)
    data[:, 40] = np.clip(data[:, 40], 90, 180)
    
    # 舒张压
    data[:, 41] = np.random.normal(80, 10, n_patients)
    data[:, 41] = np.clip(data[:, 41], 60, 120)
    
    # 心率
    data[:, 42] = np.random.normal(70, 10, n_patients)
    data[:, 42] = np.clip(data[:, 42], 50, 100)
    
    # 血糖
    data[:, 43] = np.random.normal(95, 15, n_patients)
    data[:, 43] = np.clip(data[:, 43], 70, 200)
    
    # 总胆固醇
    data[:, 44] = np.random.normal(190, 30, n_patients)
    data[:, 44] = np.clip(data[:, 44], 120, 300)
    
    # HDL胆固醇
    data[:, 45] = np.random.normal(50, 12, n_patients)
    data[:, 45] = np.clip(data[:, 45], 30, 80)
    
    # LDL胆固醇
    data[:, 46] = np.random.normal(110, 25, n_patients)
    data[:, 46] = np.clip(data[:, 46], 70, 180)
    
    # 甘油三酯
    data[:, 47] = np.random.normal(120, 40, n_patients)
    data[:, 47] = np.clip(data[:, 47], 50, 300)
    
    # 血红蛋白
    data[:, 48] = np.random.normal(14, 2, n_patients)
    data[:, 48] = np.clip(data[:, 48], 10, 18)
    
    # 白细胞计数
    data[:, 49] = np.random.normal(6.5, 1.5, n_patients)
    data[:, 49] = np.clip(data[:, 49], 3, 12)
    
    return data

def generate_cardiovascular_risk_labels(data):
    """基于患者特征生成心血管疾病风险标签"""
    
    n_patients = data.shape[0]
    risk_scores = np.zeros(n_patients)
    
    # 年龄因子
    age_factor = (data[:, 0] - 30) / 50  # 标准化年龄
    risk_scores += 0.25 * np.clip(age_factor, 0, 1)
    
    # 性别因子 (男性风险更高)
    gender_factor = (data[:, 1] == 1).astype(float)
    risk_scores += 0.1 * gender_factor
    
    # BMI因子
    bmi = data[:, 3]
    bmi_factor = np.where(bmi > 30, 0.15, 
                 np.where(bmi > 25, 0.05, 0))
    risk_scores += bmi_factor
    
    # 吸烟因子
    smoking_factor = data[:, 10] * 0.1  # 0, 0.1, 0.2
    risk_scores += smoking_factor
    
    # 高血压史
    hypertension_factor = data[:, 31] * 0.2
    risk_scores += hypertension_factor
    
    # 糖尿病史
    diabetes_factor = data[:, 30] * 0.15
    risk_scores += diabetes_factor
    
    # 心脏病家族史
    family_history_factor = data[:, 39] * 0.1
    risk_scores += family_history_factor
    
    # 血压因子
    systolic_bp = data[:, 40]
    bp_factor = np.where(systolic_bp > 140, 0.15,
                np.where(systolic_bp > 130, 0.1, 0))
    risk_scores += bp_factor
    
    # 胆固醇因子
    total_cholesterol = data[:, 44]
    chol_factor = np.where(total_cholesterol > 240, 0.1,
                  np.where(total_cholesterol > 200, 0.05, 0))
    risk_scores += chol_factor
    
    # 症状因子
    chest_pain_factor = data[:, 20] * 0.05
    shortness_breath_factor = data[:, 21] * 0.05
    risk_scores += chest_pain_factor + shortness_breath_factor
    
    # 添加随机噪声
    risk_scores += np.random.normal(0, 0.1, n_patients)
    
    # 转换为二元标签 (0: 低风险, 1: 高风险)
    threshold = np.percentile(risk_scores, 70)  # 30%的患者为高风险
    labels = (risk_scores > threshold).astype(int)
    
    return labels

def create_adaptivefs_dataset(n_patients=1000, save_dir='./Data'):
    """创建完整的AdaptiveFS数据集"""
    
    print(f"创建AdaptiveFS数据集，患者数量: {n_patients}")
    
    # 1. 创建问题名称
    question_names = create_medical_question_names()
    print(f"✓ 创建了 {len(question_names)} 个问题名称")
    
    # 2. 生成患者数据
    data = generate_realistic_medical_data(n_patients)
    print(f"✓ 生成了 {data.shape[0]} 个患者的数据，每个患者 {data.shape[1]} 个特征")
    print(f"  数据范围: [{data.min():.2f}, {data.max():.2f}]")
    
    # 3. 生成标签
    labels = generate_cardiovascular_risk_labels(data)
    print(f"✓ 生成了标签，分布: 低风险={np.sum(labels==0)}, 高风险={np.sum(labels==1)}")
    
    # 4. 数据预处理（AdaptiveFS需要[-1, 1]范围的数据）
    # 注意：某些特征（如二元变量）不需要标准化
    processed_data = data.copy()
    
    # 对连续变量进行MinMax标准化到[-1, 1]
    continuous_features = [0, 3, 4, 5, 6, 7, 8, 9,  # 人口统计学连续变量
                          12, 13, 14, 15, 16, 17, 18, 19,  # 生活方式连续变量
                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49]  # 生理指标
    
    for feat_idx in continuous_features:
        feat_data = processed_data[:, feat_idx]
        feat_min, feat_max = feat_data.min(), feat_data.max()
        if feat_max > feat_min:  # 避免除零
            processed_data[:, feat_idx] = (feat_data - feat_min) / (feat_max - feat_min) * 2 - 1
    
    # 对二元变量和分类变量进行适当处理
    binary_categorical_features = [1, 2, 10, 11] + list(range(20, 40))  # 性别、种族、吸烟、饮酒、症状、病史
    for feat_idx in binary_categorical_features:
        feat_data = processed_data[:, feat_idx]
        feat_max = feat_data.max()
        if feat_max > 0:
            processed_data[:, feat_idx] = feat_data / feat_max * 2 - 1
    
    print(f"✓ 数据预处理完成，范围: [{processed_data.min():.2f}, {processed_data.max():.2f}]")
    
    # 5. 保存文件
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存主数据文件
    np.save(os.path.join(save_dir, 'small_data50.npy'), processed_data.astype(np.float32))
    print(f"✓ 保存 small_data50.npy: {processed_data.shape}")
    
    # 保存问题名称
    np.save(os.path.join(save_dir, 'names_small50.npy'), question_names)
    print(f"✓ 保存 names_small50.npy: {question_names.shape}")
    
    # 保存标签
    np.save(os.path.join(save_dir, 'labels.npy'), labels.astype(np.int32))
    print(f"✓ 保存 labels.npy: {labels.shape}")
    
    print(f"\n✅ AdaptiveFS数据集创建完成！")
    print(f"📁 保存位置: {save_dir}")
    print(f"📊 数据统计:")
    print(f"   - 患者数量: {n_patients}")
    print(f"   - 特征数量: 50")
    print(f"   - 标签分布: {np.bincount(labels)}")
    print(f"   - 数据类型: float32")
    print(f"   - 数据范围: [{processed_data.min():.3f}, {processed_data.max():.3f}]")
    
    return processed_data, labels, question_names

if __name__ == "__main__":
    # 创建AdaptiveFS数据集
    data, labels, names = create_adaptivefs_dataset(n_patients=1500)
    
    # 验证数据加载
    print("\n🔍 验证数据集...")
    try:
        # 测试是否能用AdaptiveFS的utils加载
        import sys
        sys.path.append('.')
        from utils import load_data
        
        X, y, question_names, class_names, scaler = load_data(122)
        print(f"✅ AdaptiveFS成功加载数据: {X.shape}, 标签: {len(y)}")
        print(f"   问题名称示例: {question_names[:5]}")
        print(f"   类别名称: {class_names}")
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        
    print("\n🎉 数据集创建和验证完成！") 