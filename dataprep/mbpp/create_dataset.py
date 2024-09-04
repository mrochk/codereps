import ast
from collections import deque
from datasets import load_dataset, concatenate_datasets
import funcskeleton
from funcskeleton import SkeletonEncoder, SkeletonSerializer
from huggingface_hub import login

def get_ast_nodes_dfs(tree:ast.AST):
    """
    Get the list of nodes in an AST, as a depth-first traversal.
    """
    def dfs(node, queue=deque()):
        queue.append(type(node).__name__.upper())

        for child in ast.iter_child_nodes(node):
            queue = dfs(child, queue)

        return queue

    return '_'.join(dfs(tree))

def filter(sample:dict):
    """
    Function used to filter the dataset, that is remove nested functions etc.
    """
    function = sample['code']
    A = SkeletonEncoder.function_sanity_check(function)
    B = '*' not in function # no *args or **kwargs
    return A and B 

if __name__ == '__main__':

    mbpp = load_dataset('google-research-datasets/mbpp')

    mbpp['train'] = concatenate_datasets([mbpp['train'], mbpp['test'], mbpp['prompt']])
    del mbpp['test']; del mbpp['prompt']

    mbpp['train'] = mbpp['train'].remove_columns(['test_setup_code', 'challenge_test_list'])
    mbpp['validation'] = mbpp['validation'].remove_columns(['test_setup_code', 'challenge_test_list'])

    print(f"before: {len(mbpp['train'])}")
    print(f"before: {len(mbpp['validation'])}")

    mbpp['train'] = mbpp['train'].filter(lambda _: filter(_))
    mbpp['validation'] = mbpp['validation'].filter(lambda _: filter(_))

    print(f"after: {len(mbpp['train'])}")
    print(f"after: {len(mbpp['validation'])}")

    def f(sample):
        code = sample['code']
        tree = ast.parse(code)
        AST = get_ast_nodes_dfs(tree)
        CFG = SkeletonEncoder.from_single_functions([code])[0]
        nparams = funcskeleton.utils.n_params(tree)
        cfg = SkeletonSerializer.serialize_function_separators_numbered(CFG[1], nparams)
        sample['ast'] = AST
        sample['cfg'] = cfg
        return sample

    mbpp['train'] = mbpp['train'].map(f)
    mbpp['validation'] = mbpp['validation'].map(f)

    login()

    mbpp.push_to_hub("mbpp_ast_cfg")