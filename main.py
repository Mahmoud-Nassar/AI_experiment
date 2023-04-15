from classifier import ClassifierCommittee as CC

CSV_READ_PATH = 'dataSets\\water_potability.csv'
ATTRIBUTES = ['ph',
              'Hardness',
              'Solids',
              'Chloramines',
              'Sulfate',
              'Conductivity',
              'Organic_carbon',
              'Trihalomethanes',
              'Turbidity']
CLASSIFICATION_FIELD = ['Potability']

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    classifier = CC(CSV_READ_PATH,ATTRIBUTES,CLASSIFICATION_FIELD)
    classifier.experiment()
