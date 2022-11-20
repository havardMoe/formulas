import math
import numpy as np

def WLM(numb_of_entities,base,numb_in_entity1,numb_in_entity_2,intersection_1_2):

    a1 = max([numb_in_entity1, numb_in_entity_2])
    a2 = intersection_1_2
    a3 = numb_of_entities
    a4 = min([numb_in_entity1, numb_in_entity_2])

    args = [a1, a2, a3, a4]
    args = [math.log(a, base) if a > 0 else 0 for a in args]

    
    return 1 - ((args[0] - args[1])
              / (args[2] - args[3])
    )


def main():
    scores = [
        WLM(numb_of_entities=6, base=2, numb_in_entity1=3, numb_in_entity_2=3, intersection_1_2=3),
        WLM(numb_of_entities=6, base=2, numb_in_entity1=3, numb_in_entity_2=1, intersection_1_2=0),
        WLM(numb_of_entities=6, base=2, numb_in_entity1=3, numb_in_entity_2=3, intersection_1_2=2),
        WLM(numb_of_entities=6, base=2, numb_in_entity1=3, numb_in_entity_2=2, intersection_1_2=1),
        WLM(numb_of_entities=6, base=2, numb_in_entity1=3, numb_in_entity_2=3, intersection_1_2=2),
        WLM(numb_of_entities=6, base=2, numb_in_entity1=3, numb_in_entity_2=0, intersection_1_2=0),
        ]
    print(scores)
    print(np.argmax(scores))

if __name__ == '__main__':
    main()
