from collections        import deque
from datasets           import load_dataset, Dataset
from funcskeleton       import SkeletonEncoder, SkeletonSerializer
import funcskeleton.utils
import ast

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
    function = sample['func_code_string']
    A = SkeletonEncoder.function_sanity_check(function)
    B = '*' not in function # no *args or **kwargs
    return A and B 

def process_split(split, name, n_processes):
    data = []

    print(f'Loading {name} split...', flush=True)

    functions = [_['func_code_string'] for _ in split]

    srcs_cfgs = SkeletonEncoder.from_single_functions_parallel(
        functions=functions, verbose=True, n_processes=n_processes,
    )

    for src, cfg in srcs_cfgs:
        tree    = ast.parse(src)
        nparams = funcskeleton.utils.n_params(tree)
        cfg_str = SkeletonSerializer.serialize_function_separators_numbered(cfg, nparams)
        ast_str = get_ast_nodes_dfs(tree)
        data.append({'src':src, 'cfg':cfg_str, 'ast':ast_str})

    print(f'Writing in {name}.json ...', flush=True)
    hface = Dataset.from_list(data)
    hface.to_json(f'{name}.json')

if __name__ == '__main__':

    # load CodeSearchNet
    code_search_net = load_dataset(
        'code_search_net', 'python',
        trust_remote_code=True,
    )

    print(f'Before(train): {code_search_net["train"].num_rows}')
    print(f'Before(test):  {code_search_net["test"].num_rows}')
    print(f'Before(validation): {code_search_net["validation"].num_rows}')

    # apply filter
    code_search_net['train'] = code_search_net['train'].filter(lambda _: filter(_))
    code_search_net['test']  = code_search_net['test'].filter(lambda _: filter(_))
    code_search_net['validation'] = code_search_net['validation'].filter(lambda _: filter(_))

    print(f'After filtering (train): {code_search_net["train"].num_rows}')
    print(f'After filtering (test):  {code_search_net["test"].num_rows}')
    print(f'After filtering (validation): {code_search_net["validation"].num_rows}')

    csn_train = code_search_net['train']
    csn_test  = code_search_net['test']
    csn_val   = code_search_net['validation']
    
    splits = [csn_test, csn_val, csn_train]
    split_names = ['test', 'validation', 'train']

    for split, name in zip(splits, split_names):
        process_split(split, name, n_processes=20)