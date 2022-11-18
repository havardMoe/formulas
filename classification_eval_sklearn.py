from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_metrics(actual, pred):
    cm = confusion_matrix(actual, pred)
    metrics = {
        'accuracy': accuracy_score(actual, pred),
        'precision': precision_score(actual, pred),
        'recall': recall_score(actual, pred),
        'f1': f1_score(actual, pred),
    }

    print('-' * 69)

    print('Input Confusion Matrix:')
    print(cm)

    print('-' * 30, 'metrics', '-' * 30)
    for k, v in metrics.items():
        print(k, v, sep='\t')

    print('-' * 69)
    
    return metrics


def main():
    actual = [0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    pred =   [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
    get_metrics(actual, pred)


if __name__ == '__main__':
    main()
