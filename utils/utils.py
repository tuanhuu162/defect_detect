import pandas as pd
import os
from shutil import copy
from javalang.parse import parse
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
import re

SPECIAL = ["<UNK>", "<BOF>", "<EOF>"]

def _prepare_raw_data():
    DATA_PATH = "elasticsearch"
    data_results = pd.DataFrame(columns=['ID','Name','LongName','Parent','McCC','CLOC','LLOC',
                                         'Number of previous fixes','Number of developer commits','Number of committers',
                                         'Number of previous modifications','Number of bugs'])
    new_path = []
    for id, file in enumerate(os.listdir(DATA_PATH)):
        source_path = DATA_PATH + "/" + file
        data = pd.read_csv(source_path + "/elasticsearch-File.csv")
        data_results = data_results.append(data)
        if not os.path.exists(source_path + "/data"):
            os.mkdir(source_path + "/data")
        if not os.path.exists("elastic"):
            os.mkdir("elastic")
        if not os.path.exists("elastic/data" + str(id)):
            os.mkdir("elastic/data" + str(id))
        for i in range(len(data)):
            file_path = data.loc[i, 'LongName']
            file_name = data.loc[i, 'Name']
            dst = os.path.join("elastic/data" + str(id))
            copy(source_path + "/" + file_path, dst)
            new_path.append(dst + "/" + file_name)
    # print(data_results)
    data_results['new_path'] = new_path
    data_results.to_csv('metadata.csv')
    print("Finish!!!!")

def preproces(filename, vocab, output_path):
    test = """
public class HelloWorld {

    public static void test(){
        int x = 0;
        if ( ! stack.empty() ){
            while ( x < 10 && x > 4){
                int y ;
                y = stack.pop();
                x++;
                x += 1;
                x = x + 1;
            }
        }
    }
    
    public static void main(String[] args) {
        // Prints "Hello, World" to the terminal window.
        System.out.println("Hello, World");
        test();
    }
}
"""
    with open(filename) as file:
        node = parse(file.read().strip())
        start = Node("<BOF>")
        traveler(node, start)
        for pre, fill, n in RenderTree(start):
            vocab.append(n.name)
    basename = os.path.basename(filename).split(".")[0]
    exporter = JsonExporter()
    with open(os.path.join(output_path, basename + ".json"), "w") as file:
        file.write(exporter.export(start))


def traveler(parse_node, tree_node):
    catch_type = r"^\w+(?=\()"
    catch_literial = r"(?<=(^Literal\())(.*value=(\w+))" # group3
    catch_member = r"(?<=(^MemberReference\(member=))(\w+), postfix_operators=\[\'?([+-]{0,2})\'?\], prefix_operators=\[\'?([+-]{0,2})\'?\]" # group 2 3 4
    catch_compare = r"(?<=(^BinaryOperation\())((.*)operator=([<>=!&|+-\/%^]+))(?=\)$)" # group 4
    catch_assignment = r"(?<=(^Assignment\())((.*)type\=([<>=!&|+-\/%^]+))" # group 4
    match = re.match(catch_type, parse_node.__str__())
    if match:
        type_n = match.group(0)
    else:
        return
    new_node = Node(type_n, parent=tree_node)
    list_child = []
    for i, n in enumerate(parse_node.children):
        if isinstance(n, list):
            for instance in n:
                if re.match(r"\<class \'javalang\.tree\..*\'\>",str(type(instance))):
                    list_child.append(instance)
        if re.match(r"\<class \'javalang\.tree\..*\'\>",str(type(n))):
            list_child.append(n)
    if len(list_child) == 0:
        return
    else:
        # print(list_child)
        for child in list_child:
            traveler(child, new_node)
        
if __name__=="__main__":
    preproces('../elastic/data0/ElasticSearchException.java', [], ".")