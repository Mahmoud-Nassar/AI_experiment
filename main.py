from classifier import ClassifierCommittee as CC

if __name__ == '__main__':

    dataSetsdataSetSpecifications = [
        # ['dataSets\\water_potability.csv', ['ph',
        #                                     'Hardness',
        #                                     'Solids',
        #                                     'Chloramines',
        #                                     'Sulfate',
        #                                     'Conductivity',
        #                                     'Organic_carbon',
        #                                     'Trihalomethanes',
        #                                     'Turbidity'], ['Potability']],
        # ['dataSets\\diabetes.csv', ['patient_number',
        #                              'cholesterol',
        #                              'glucose',
        #                              'hdl_chol',
        #                              'chol_hdl_ratio',
        #                              'age',
        #                              'gender',
        #                              'height',
        #                              'weight',
        #                              'bmi',
        #                              'systolic_bp',
        #                              'diastolic_bp',
        #                              'waist',
        #                              'hip',
        #                              'waist_hip_ratio'], ['diabetes']],
        ['dataSets\\heart_attack.csv', ['age',
                                        'sex',
                                        'cp',
                                        'trtbps',
                                        'chol',
                                        'fbs',
                                        'restecg',
                                        'thalachh',
                                        'exng',
                                        'oldpeak',
                                        'slp',
                                        'caa',
                                        'thall'], ['output']]
    ]

    for dataSetSpecification in dataSetsdataSetSpecifications:
        classifier = CC(dataSetSpecification[0], dataSetSpecification[1], dataSetSpecification[2])
    result = classifier.experiment()
    dataBaseName = dataSetSpecification[0].split("\\")[-1].replace(".csv", "")
    print(("{} : \n"
           "Precision majority approach = {}% \n"
           "Precision random choosing approach = {}% \n"
           "Precision best trainig Precision approach = {}% \n"
           "Precision of distance approache = {}% \n").
          format(dataBaseName,result[0], result[1], result[2], result[3]))
