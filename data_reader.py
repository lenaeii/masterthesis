import pandas as pd # type: ignore # type: ignore

# Read the preprocessed xlsx file
def read_and_transform(file_path):
    df = pd.read_excel(file_path)
    data = []
    for index, row in df.iterrows():
        case_id = str(row['case_id'])
        activities = row['activities'].split(', ')
        data.append({"case_id": case_id, "activities": activities})
    
    return data