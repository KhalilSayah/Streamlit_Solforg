from models import CategoricalCriteria, CategoricalCriteriaPart, NumericCriteria, NumericCriteriaPart

def get_criteria_list():
    return [
        NumericCriteria(
            criteria_type = 'primary',
            label="Joining Time",
            criteria_parts=[
                NumericCriteriaPart(label="< 10", min_value=0, max_value=10, score=2),
                NumericCriteriaPart(label="< 25", min_value=10, max_value=25, score=1.8),
                NumericCriteriaPart(label="< 50", min_value=25, max_value=50, score=1.5),
                NumericCriteriaPart(label="< 100", min_value=50, max_value=100, score=2.5),
                NumericCriteriaPart(label="< 150", min_value=100, max_value=150, score=1),
            ]
        ),
        CategoricalCriteria(
            criteria_type = 'primary',
            label="Seniority Level",
            criteria_parts=[
                CategoricalCriteriaPart(label="Entry Level", score=1),
                CategoricalCriteriaPart(label="Junior", score=1.5),
                CategoricalCriteriaPart(label="Mid-Level", score=2),
                CategoricalCriteriaPart(label="Senior", score=2.5),
                CategoricalCriteriaPart(label="Lead/Principal", score=3),
                CategoricalCriteriaPart(label="Manager", score=2.5),
                CategoricalCriteriaPart(label="Division", score=3),
            ]
        ),
        CategoricalCriteria(
            criteria_type = 'primary',
            label="Role Importance",
            criteria_parts=[
                CategoricalCriteriaPart(label="Engineering", score=1),
                CategoricalCriteriaPart(label="Business Dev", score=1.5),
                CategoricalCriteriaPart(label="Legal", score=2),
                CategoricalCriteriaPart(label="Marketing", score=2.5),
                CategoricalCriteriaPart(label="Operations", score=3),
                CategoricalCriteriaPart(label="Support", score=1),
                
            ]
            
        ),
        CategoricalCriteria(
            criteria_type = 'primary',
            label="Salary Compensation",
            criteria_parts=[
                CategoricalCriteriaPart(label="< 100k", score=1.2),
                CategoricalCriteriaPart(label="100-150k", score=1.1),
                CategoricalCriteriaPart(label="150-200k", score=1),
                CategoricalCriteriaPart(label="200k-250k", score=0.9),
                CategoricalCriteriaPart(label="> 250k", score=0.8)
                
            ]
            
        ),
        CategoricalCriteria(
            criteria_type = 'bonus',
            label="Bonus - Individual Performance",
            criteria_parts=[
                CategoricalCriteriaPart(label="Needs Improvement", score=2),
                CategoricalCriteriaPart(label="Meets Excpectations", score=1.8),
                CategoricalCriteriaPart(label="Exceeds Expectations", score=1.5),
                CategoricalCriteriaPart(label="Outstanding", score=1.2),
                CategoricalCriteriaPart(label="Exceptional", score=1)
                
            ]
            
        ),
        CategoricalCriteria(
            criteria_type = 'bonus',
            label="Bonus - Project Impact",
            criteria_parts=[
                CategoricalCriteriaPart(label="Standard", score=2),
                CategoricalCriteriaPart(label="High Impact", score=1.8),
                CategoricalCriteriaPart(label="Critical Success", score=1.5),
                
            ]
            
        ),
        CategoricalCriteria(
            criteria_type = 'bonus',
            label="Bonus - Innovation Contribution",
            criteria_parts=[
                CategoricalCriteriaPart(label="Standard", score=1),
                CategoricalCriteriaPart(label="Notable Innovation", score=1.15),
                CategoricalCriteriaPart(label="Significant Innovation", score=1.3),
                
            ]
            
        ),
        CategoricalCriteria(
            criteria_type = 'bonus',
            label="Bonus - Tenure Adjustment",
            criteria_parts=[
                CategoricalCriteriaPart(label="0-2 Years", score=1),
                CategoricalCriteriaPart(label="2-4 Years", score=1.1),
                CategoricalCriteriaPart(label="4+ Years", score=1.2),
                
            ]
            
        ),
        CategoricalCriteria(
            criteria_type = 'bonus',
            label="Bonus - Composition Factor",
            criteria_parts=[
                CategoricalCriteriaPart(label="100% Toekns", score=1),
                CategoricalCriteriaPart(label="75% Tokens", score=0.85),
                CategoricalCriteriaPart(label="50% Tokens", score=0.7),
                CategoricalCriteriaPart(label="25% Tokens", score=0.55),
                
            ]
            
        ),

    ]
