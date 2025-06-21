"""
简化版AdaptiveFS数据集生成脚本
不使用sklearn依赖，纯numpy实现
"""

import numpy as np
import os

def create_medical_question_names():
    """创建50个医学问卷问题名称"""
    
    question_names = [
        # 基本人口统计学信息 (0-9)
        'age_p', 'sex', 'hiscodi32', 'bmi', 'education_level',
        'income_level', 'marital_status', 'employment_status', 'insurance_status', 'region_code',
        
        # 生活方式因素 (10-19)
        'smoking_status', 'alcohol_consumption', 'exercise_frequency', 'diet_quality', 'sleep_hours',
        'stress_level', 'caffeine_intake', 'physical_activity', 'sedentary_time', 'social_support',
        
        # 症状和主诉 (20-29)
        'chest_pain', 'shortness_of_breath', 'fatigue', 'dizziness', 'headache',
        'nausea', 'abdominal_pain', 'joint_pain', 'back_pain', 'sleep_problems',
        
        # 既往病史 (30-39)
        'diabetes_history', 'hypertension_history', 'heart_disease_history', 'stroke_history', 'cancer_history',
        'kidney_disease_history', 'liver_disease_history', 'lung_disease_history', 'mental_health_history', 'family_heart_disease',
        
        # 实验室检查和生理指标 (40-49)
        'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'blood_glucose', 'cholesterol_total',
        'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides', 'hemoglobin', 'white_blood_cell_count'
    ]
    
    return np.array(question_names, dtype='<U32')

def generate_medical_data(n_patients=1500):
    """生成医学数据"""
    
    np.random.seed(42)
    data = np.zeros((n_patients, 50))
    
    # 基本信息
    data[:, 0] = np.clip(np.random.normal(45, 15, n_patients), 18, 85)  # 年龄
    data[:, 1] = np.random.choice([1, 2], n_patients)  # 性别
    data[:, 2] = np.random.choice([0, 1], n_patients)  # 种族
    data[:, 3] = np.clip(np.random.normal(25, 5, n_patients), 15, 50)  # BMI
    
    # 其他特征
    for i in range(4, 50):
        if i in [10, 11] + list(range(20, 40)):  # 分类特征
            data[:, i] = np.random.randint(0, 4, n_patients)
        else:  # 连续特征
            data[:, i] = np.random.normal(0, 1, n_patients)
    
    return data

def generate_labels(data):
    """生成标签"""
    
    # 基于年龄、BMI、症状等生成心血管风险标签
    risk_score = (
        (data[:, 0] - 30) / 50 * 0.3 +  # 年龄
        (data[:, 3] - 20) / 30 * 0.2 +  # BMI  
        np.mean(data[:, 20:30], axis=1) * 0.2 +  # 症状
        np.mean(data[:, 30:40], axis=1) * 0.3    # 病史
    )
    
    # 添加噪声
    risk_score += np.random.normal(0, 0.1, len(data))
    
    # 转为二元标签
    threshold = np.percentile(risk_score, 70)
    labels = (risk_score > threshold).astype(int)
    
    return labels

def normalize_data(data):
    """数据标准化到[-1, 1]范围"""
    
    normalized_data = data.copy()
    
    for i in range(data.shape[1]):
        col_data = data[:, i]
        col_min, col_max = col_data.min(), col_data.max()
        
        if col_max > col_min:
            normalized_data[:, i] = (col_data - col_min) / (col_max - col_min) * 2 - 1
        else:
            normalized_data[:, i] = 0
    
    return normalized_data

def create_adaptivefs_dataset():
    """创建完整的AdaptiveFS数据集"""
    
    print("创建AdaptiveFS数据集...")
    
    # 生成数据
    n_patients = 1500
    data = generate_medical_data(n_patients)
    labels = generate_labels(data)
    question_names = create_medical_question_names()
    
    # 数据标准化
    normalized_data = normalize_data(data)
    
    print(f"✓ 生成了 {n_patients} 个患者，50个特征")
    print(f"✓ 标签分布: 低风险={np.sum(labels==0)}, 高风险={np.sum(labels==1)}")
    print(f"✓ 数据范围: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
    
    # 保存文件
    os.makedirs('./Data', exist_ok=True)
    
    np.save('./Data/small_data50.npy', normalized_data.astype(np.float32))
    np.save('./Data/names_small50.npy', question_names)
    np.save('./Data/labels.npy', labels.astype(np.int32))
    
    print("✅ 数据集创建完成！")
    print("📁 文件保存在 ./Data/ 目录")
    print("   - small_data50.npy")
    print("   - names_small50.npy")
    print("   - labels.npy")
    
    return normalized_data, labels, question_names

if __name__ == "__main__":
    data, labels, names = create_adaptivefs_dataset()
    
    # 验证数据加载
    print("\n🔍 验证AdaptiveFS数据加载...")
    try:
        import utils
        X, y, question_names, class_names, scaler = utils.load_data(122)
        print(f"✅ AdaptiveFS成功加载数据！")
        print(f"   数据形状: {X.shape}")
        print(f"   标签数量: {len(y)}")
        print(f"   问题名称示例: {question_names[:3]}")
        print(f"   类别名称: {class_names}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
    
    print("\n🎉 AdaptiveFS数据集准备完成！") 