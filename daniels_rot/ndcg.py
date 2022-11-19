
import math
from typing import List

system_rankings = {
    "q1": [[1,3,2,4,5],[4,5,2,3,1],[2,3,1,4,5]],
    "q2": [[1,3,5,4,2],[5,4,1,2,3],[2,4,1,3,5]],
    "q3": [[4,2,3,1,5],[3,5,2,4,1],[4,3,5,1,2]]
}

ground_truth = {
    "q1": {1: 3, 2: 1},
    "q2": {2:1,4:1},
    "q3": {2:3,3:3,4:1}
}


def dcg(relevances: List[int], k: int) -> float:

    return relevances[0] + sum(
        rel / math.log(i + 2, 2) 
         for i, rel in enumerate(relevances[1:k])
    )

def ndcg(system_ranking: List[int], ground_truth: List[int], k:int = 10) -> float:
    relevances = [ground_truth.get(rank,0) for rank in system_ranking]
    relevances_ideal = sorted(ground_truth.values(), reverse=True)
    return dcg(relevances, k) / dcg(relevances_ideal, k)


A = ndcg(system_ranking=system_rankings['q3'][0],ground_truth=ground_truth['q3'],k=5)
B = ndcg(system_ranking=system_rankings['q3'][1],ground_truth=ground_truth['q3'],k=5)
C = ndcg(system_ranking=system_rankings['q3'][2],ground_truth=ground_truth['q3'],k=5)

print(f'A: {A}\nB: {B}\nC: {C}')


A_avg = sum([ndcg(system_ranking=system_rankings[q][0],ground_truth=ground_truth[q],k=5)for q in system_rankings.keys()]) /3
B_avg = sum([ndcg(system_ranking=system_rankings[q][1],ground_truth=ground_truth[q],k=5) for q in system_rankings.keys()])/3
C_avg = sum([ndcg(system_ranking=system_rankings[q][2],ground_truth=ground_truth[q],k=5) for q in system_rankings.keys()]) /3

print(f'A: {A_avg}\nB: {B_avg}\nC: {C_avg}')