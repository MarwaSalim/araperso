def classifaction_report_csv(report,confusionMatrix,Path):
    import pandas as pd
    report_data = []
    lines = report.split('\n')
    i=0
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        row['0'] = confusionMatrix[i][0]
        row['1'] = confusionMatrix[i][1]
        report_data.append(row)
        i+=1
    row = {}
    row_data = lines[-2].split('      ')
    row['class'] = row_data[0]
    row['precision'] = float(row_data[1])
    row['recall'] = float(row_data[2])
    row['f1_score'] = float(row_data[3])
    row['support'] = float(row_data[4])
    report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(Path+".csv", index = False)