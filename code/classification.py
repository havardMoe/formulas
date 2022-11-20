def one_versus_rest_classification(preds):
    votes = [0] * len(preds)
    for i, pred in enumerate(preds):
        votes[i] += pred
        for j in range(len(votes)):
            if i != j:
                votes[j] += 1 - pred
    return votes

def test_class():
    preds = [0, 1, 0, 0]
    assert one_versus_rest_classification(preds) == [2, 4, 2, 2]

def main():
    test_class()

if __name__ == '__main__':
    main()