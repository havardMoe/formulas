import math

def WLM(numb_of_entities,base,numb_in_entity1,numb_in_entity_2,intersection_1_2):


    return (1-((math.log(max([numb_in_entity1,numb_in_entity_2]),base)-math.log(intersection_1_2,base))/(math.log(numb_of_entities,base)-math.log(min([numb_in_entity1,numb_in_entity_2]),base))))


print(WLM(numb_of_entities=6,base=2,numb_in_entity1=1,numb_in_entity_2=2,intersection_1_2=1))