from classifier import ClassifierCommittee as CC

if __name__ == '__main__':

    dataSetSpecifications = [
        ['dataSets\\water_potability.csv', ['ph',
                                            'Hardness',
                                            'Solids',
                                            'Chloramines',
                                            'Sulfate',
                                            'Conductivity',
                                            'Organic_carbon',
                                            'Trihalomethanes',
                                            'Turbidity'], ['Potability']],
        ['dataSets\\voice_gender_Recognition.csv', ['meanfreq',
                                                    'sd',
                                                    'median',
                                                    'Q25',
                                                    'Q75',
                                                    'IQR',
                                                    'skew',
                                                    'kurt',
                                                    'sp_ent',
                                                    'sfm',
                                                    'mode',
                                                    'centroid',
                                                    'meanfun',
                                                    'minfun',
                                                    'maxfun',
                                                    'meandom',
                                                    'mindom',
                                                    'maxdom',
                                                    'dfrange',
                                                    'modindx'], ['label']],
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

    for dataSetSpecification in dataSetSpecifications:
        classifier = CC(dataSetSpecification[0], dataSetSpecification[1], dataSetSpecification[2])
        result = classifier.experiment()
        dataBaseName = dataSetSpecification[0].split("\\")[-1].replace(".csv", "")
        with open('output.txt', 'a') as f:
            f.write(("{} : \n"
                     "Precision majority approach = {}% \n"
                     "Precision random choosing approach = {}% \n"
                     "Precision best training Precision approach = {}% \n"
                     "Precision of distance approach = {}% \n").
                    format(dataBaseName, result[0], result[1], result[2], result[3]))
