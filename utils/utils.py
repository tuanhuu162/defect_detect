import pandas as pd
import os
from shutil import copy
from javalang.parse import parse
from javalang.parser import JavaSyntaxError
from javalang.tokenizer import tokenize
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import JsonExporter
from anytree.walker import Walker
from copy import deepcopy
import re
from lxml import etree
import json

SPECIAL = ["<UNK>", "<BOF>", "<EOF>"]
DATA_PATH = "../"
OUTDIR = "../json_data"


def _prepare_raw_data():
    DATA_PATH = "elasticsearch"
    data_results = pd.DataFrame(columns=['ID', 'Name', 'LongName', 'Parent', 'McCC', 'CLOC', 'LLOC',
                                         'Number of previous fixes', 'Number of developer commits',
                                         'Number of committers',
                                         'Number of previous modifications', 'Number of bugs'])
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
import java.util.*;

public class Example1{
    public static void main(String[] args){
        Stack<Integer> stack = new Stack<>();
        int x = 0;
        if (!stack.empty()){
            while ( x < 10){
                int y ;
                y = stack . pop ( ) ;
                x++;
            }
        }
    }
}

"""
    with open(filename, encoding='utf-8') as file:
        try:
            node = parse(file.read().strip())
            start = Node("<BOF>")
            traveler(node, start, [], [])
            for pre, fill, n in RenderTree(start):
                vocab.append(n.name)
        except Exception or AttributeError:
            print(filename)
            return ''
    # basename = os.path.basename(filename).split(".")[0]
    # foldername = os.path.dirname(filename).split("/")[1]
    node_1 = parse(test)
    start_1 = Node("<BOF>")
    # print(node_1)
    traveler(node_1, start_1, [], [])
    exporter = JsonExporter()
    for path, n in node_1:
        print(n)
    for pre, fill, b in RenderTree(start_1):
        print(pre, fill, b.name)
    print(exporter.export(start_1))
    # new_path = output_path + "/" + foldername + basename + ".json"
    # with open(new_path, "w") as file:
    #     file.write(exporter.export(start))
    # return new_path
    return exporter.export(start)


def preprocess_ver2(xml_file, vocab):
    with open(xml_file) as file:
        xmlstring = file.read()
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(xmlstring, parser=parser)
    print("Extracting xml file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    data = pd.read_csv(os.path.join(DATA_PATH, "Benchmark_v1.1_Scorecard_for_FindBugs.csv"))
    list_label = {}
    for i in range(len(data)):
        if data.loc[i, ' real vulnerability'] == " true" and data.loc[i, ' pass/fail'] == " pass":  # true positive
            list_label[data.loc[i, '# test name']] = 1
        elif data.loc[i, ' real vulnerability'] == " false" and data.loc[i, ' pass/fail'] == " fail":  # false positive
            list_label[data.loc[i, '# test name']] = 0
        else:
            list_label[data.loc[i, '# test name']] = 2

    list_data = {
        "bug": [],
        "method": [],
        "label": []
    }

    with open(xml_file, 'rb') as file:
        xmlstring = file.read()
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(xmlstring, parser=parser)
    print("Extracting xml file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # file.write("bug|method|label\n")
    for ele in tree:
        if ele.tag == "BugInstance":
            bug_instances = []
            # class_attrib = []
            isTest = False
            virtual_data = {
                "class": [],
                "field": [],
                "method": []
            }
            for child in ele:
                # class_name = ""
                if child.tag == "Class":
                    if "testcode." in child.attrib['classname']:
                        isTest = True
                        for c in child:
                            if c.tag == "SourceLine":
                                # class_attrib = [c.attrib['sourcepath'], int(c.attrib['start']), int(c.attrib['end'])]
                                start_class = int(c.attrib['start'])
                                end_class = int(c.attrib['end'])
                                # if len(virtual_data["class"]) > 0:
                                #     if end_class - start_class > virtual_data["class"][2] - virtual_data["class"][1]:
                                #         virtual_data["class"] = [c.attrib['sourcepath'], start_class, end_class]
                                # else:
                                virtual_data["class"] = [c.attrib['sourcepath'], start_class, end_class]
                                # class_name = c.attrib['classname']
                if child.tag == "Method" or child.tag == "Field":
                    if "testcode." in child.attrib['classname']:
                        if child.tag == "Method":
                            for c in child:
                                if c.tag == "SourceLine":
                                    if ele.attrib['type'] in ["UC_USELESS_VOID_METHOD", "ESync_EMPTY_SYNC"]:
                                        bug_instances.append(c.attrib)
                                    # print(c.attrib)
                                    if 'start' in c.attrib:
                                        virtual_data["method"].append(
                                            (int(c.attrib['start']), int(c.attrib['end'])))
                        else:
                            virtual_data["field"].append(child.attrib['name'])
                if child.tag == "SourceLine":
                    if ele.attrib['type'] not in ["UC_USELESS_VOID_METHOD", "ESync_EMPTY_SYNC"]:
                        attrib = child.attrib
                        bug_instances.append(attrib)
            if len(bug_instances) > 0:
                # extract_data(class_attrib, bug_instances, vocab, list_data, isBug)
                if isTest:
                    extract_data(virtual_data, bug_instances, vocab, list_data, list_label)
            else:
                print("DO NOT HANDLE CLASS ERROR")
    vocab = set(vocab)
    with open(os.path.join(OUTDIR, "vocab"), 'w') as file:
        for i in vocab:
            file.write(i + "\n")
    with open(os.path.join(OUTDIR, "data_owasp.json"), 'w') as file:
        file.write(json.dumps(list_data))

# def extract_data(filename, bug_instances, vocab, data_file, list_label):
# def extract_data(class_attrib, bug_instances, vocab, list_data, label):
#     base_name = class_attrib[0].split("/")[-1].split(".")[0]
def extract_data(virtual_data, bug_instances, vocab, list_data, list_label):
    base_name = virtual_data['class'][0].split("/")[-1].split(".")[0]
    match_cmt = r"\/\/.*"
    match_class = "(\w+\.)+[A-Z]\w+"
    match_string = r"\"([^\"]+)\"|\'([^\"']+)\'"
    exporter = JsonExporter()
    start = Node("<BOF>")
    print("Extracting " + base_name + " tree!!!!!!!!!!!!!!!!!!!")

    label = list_label[base_name]
    if label == 2:
        return ''
    try:
        with open(os.path.join(DATA_PATH, virtual_data['class'][0])) as file:
            string = file.read().strip()
            node = parse(string)
            lines = [line.strip() for line in string.split("\n")]
    except AttributeError:
        print(base_name)
        return ''
    list_re = []
    for bug in bug_instances:
        start_line = int(bug['start'])
        end = int(bug['end'])
        for line in range(int(start_line) - 1, int(end)):
            clean_cmt = re.sub(match_cmt, " ", lines[line])
            clean_string, list_string = search_and_replace(match_string, clean_cmt)
            clean_class, list_class = search_and_replace(match_class, clean_string)
            clean = re.sub(r"[\s;\]\[(){}=+\-/?:!~<>.,%^&*#$|]+", " ", clean_class)
            list_clean = [i for i in clean.split(" ") if i != ""]
            list_clean.extend([i.group().replace("\"", r"\"") for i in list_string])
            list_clean.extend([i.group().replace(".", r"\.") for i in list_class])
            final_list = list_clean
            for string in list_clean:
                if len(string.split(r"\.")) >= 2:
                    final_list.extend(string.split(r"\."))
            if len(list_clean) > 0:
                list_re.append("|".join(final_list))
    # print(list_re)
    list_re = list(set(list_re))
    len_match = [0 for i in range(len(list_re))]
    traveler(node, start, list_re, len_match)
    defect_node = [[] for i in range(len(list_re))]
    max_value = [0 for i in range(len(list_re))]
    for n in PreOrderIter(start):
        if hasattr(n, "len_match"):
            for i in range(len(n.len_match)):
                if n.len_match[i] > max_value[i]:
                    max_value[i] = n.len_match[i]
                    defect_node[i] = [n]
                elif n.len_match[i] == max_value[i]:
                    defect_node[i].append(n)
    defect_node = [j for i in defect_node for j in i]
    # print(defect_node)
    for i, n in enumerate(defect_node):
        # n.regex = list_re[i]
        n.isBug = True
    method = exporter.export(start)
    for pre, fill, n in RenderTree(start):
        vocab.append(n.name)
    # # list_bug, list_method = get_branch(start, defect_node)
    bug = get_branch(defect_node)
    list_data["bug"].append(exporter.export(bug))
    list_data["method"].append(method)
    list_data["label"].append(label)

def traveler(parse_node, tree_node, list_regex, len_match):
    catch_type = r"^\w+(?=\()"
    catch_literial = r"(?<=(^Literal\())(.*value=(\w+))"  # group3
    catch_member = r"(?<=(^MemberReference\(member=))(\w+), postfix_operators=\[\'?([+-]{0,2})\'?\], prefix_operators=\[\'?([+-]{0,2})\'?\]"  # group 2 3 4
    catch_compare = r"(?<=(^BinaryOperation\())((.*)operator=([<>=!&|+-\/%^]+))(?=\)$)"  # group 4
    catch_assignment = r"(?<=(^Assignment\())((.*)type\=([<>=!&|+-\/%^]+))"  # group 4
    match = re.match(catch_type, parse_node.__str__())

    if match:
        type_n = match.group(0)
    else:
        return
    new_node = Node(type_n, parent=tree_node)
    list_child = []
    # print(parse_node)
    for i, r in enumerate(list_regex):
        for attr in parse_node.attrs:
            try:
                # print(attr, getattr(parse_node, attr))
                if isinstance(getattr(parse_node, attr), str):
                    len_match[i] += len(re.findall(r, getattr(parse_node, attr)))
            except Exception as e:
                print(r)
                raise e

    for i, n in enumerate(parse_node.children):
        if isinstance(n, list):
            for instance in n:
                if re.match(r"\<class \'javalang\.tree\..*\'\>", str(type(instance))):
                    list_child.append(instance)
        if re.match(r"\<class \'javalang\.tree\..*\'\>", str(type(n))):
            list_child.append(n)
    if len(list_child) == 0:
        # new_node.attr = [(attr, getattr(parse_node, attr)) for attr in parse_node.attrs if isinstance(getattr(parse_node, attr), str)]
        new_node.len_match = len_match
        return
    else:
        # print(list_child)
        for child in list_child:
            len_father = deepcopy(len_match)
            traveler(child, new_node, list_regex, len_father)

def search_and_replace(pattern, string):
    list_string = re.finditer(pattern, string, re.M)
    clean_string = re.sub(pattern, " ", string)
    return clean_string, list_string

def get_branch(defect_node):
    root = defect_node[0].root
    exporter = JsonExporter()

    for node in defect_node:
        for n in node.path:
            n.track = True

    for i, node in enumerate(PreOrderIter(root)):
        if not hasattr(node, 'track'):
            node.parent = None
    return root

# def make_virtual_class(class_attrib, bug_attrib):
#     source, start_class, end_class = class_attrib
def make_virtual_class(virtual_data, bug_attrib):
    source, start_class, end_class = virtual_data['class']
    print(start_class, end_class)
    virtual_class = ""
    cmt_re = r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/"
    import_re = r"(import|package).*\;"
    class_re = r"^\s*(public|private|protected|default)?\s{1,3}class\s{1,3}[\w\s\n\-\.\<\>\[\]\,]*\{"
    method_re = r"^\s*(((public|private|protected|default)\s{1,3}(static\s{1,3})?)|(static\s{1,3}(public|private|protected|default)\s{1,3}))?(\w+[\w\s\.\<\>\,\[\]\-]+)\s{1,3}\w[\w-]+\s{0,3}\([\w\s\.\<\>\,\[\]\-]*\)[\w\s\.\<\>\,\[\]\-]*\{"
    # attrib_re = "(((public|private|protected|default)\s{1,3}(static\s{1,3})?)|((static\s{1,3})(public|private|protected|default)\s{1,3}))?\w[\w\.\-<>\*\[\]\s]+\s{1,3}\w[\w-]+(\s{0,3}=.*\n{0,3}.*)?;"
    attrib_re = "^\s*((public|private|protected|default|static|final){1,3}\s{1,3})?\w[\w\.\-<>\*\[\]\s]+\s{1,3}\w[\w\.\-<>\*\[\]]+(\s{0,3}=.*\n{0,3}.*)?;"
    with open(os.path.join(DATA_PATH, source)) as file:
        string = remove_cmt(file.read(), cmt_re)
        lines = [line.strip() for line in string.split("\n")]

    list_import = "\n".join([i.group() for i in re.finditer(import_re, "\n".join(lines[:start_class]))])
    list_index = []
    list_new_index = []
    for bug in bug_attrib:
        start, end = int(bug['start']), int(bug['end'])
        has_method = False
        print(start, end)
        print(virtual_data['method'])
        if len(virtual_data['method']) > 0:
            for method in virtual_data['method']:
                if start in range(method[0], method[1] + 1) and end in range(method[0], method[1] + 1):
                    has_method = True
                    start_method, end_method = find_method_line(lines, start_class, method[0], end_class, method_re)
                    break
        if not has_method:
            start_method, end_method = find_method_line(lines, start_class, start, end_class, method_re)
        list_index.append((start_method, end_method, start, end))
    max_start_bug = max(list_index, key=lambda x: x[3])[3]
    min_start_bug = min(list_index, key=lambda x: x[3])[3]
    # print(max_start_method, min_start_method)
    list_attrib = find_class_attrib(lines[start_class - 1: max_start_bug], attrib_re)
    # print(list_attrib)
    virtual_class += list_import + "\n"
    # print("\n".join(lines[start_class - 1: start_method]))
    virtual_class += re.match(class_re, "\n".join(lines[start_class - 1: min_start_bug])).group() + "\n"
    virtual_class += "\n".join(list_attrib) + "\n"
    start_method_new = len(virtual_class.split("\n"))
    # start_bug_from_start_method = start - start_method + start_method_new - 1
    # end_bug_from_start_method = start_bug_from_start_method + end - start
    # print(start_bug_from_start_method, end_bug_from_start_method)

    for i in list_index:
        print(i)
        if i[2] in range(i[0], i[1] + 1):
            virtual_class += "\n".join(lines[i[0]: i[1]]) + "\n"
            start_bug = start_method_new - 1 + i[2] - i[0]
            end_bug = i[3] - i[2] + start_bug
        else:
            bug = [i.group() for i in re.finditer(attrib_re, "\n".join(lines[i[2] - 1:i[3]]))][0]
            # print(bug)
            for i, line in enumerate(virtual_class.split("\n")):
                if line == bug:
                    start_bug = i + 1
                    end_bug = i + 1
                    break
        list_new_index.append((start_bug, end_bug))
        start_method_new = len(virtual_class.split("\n"))

    virtual_class += "\n}"
    # print(virtual_class)
    return virtual_class, list_new_index

def find_parentless_line(string):
    open_p = 0
    line_start = 0
    line_end = 0
    z = False
    for i, line in enumerate(string):
        # print(open_p, i)
        if z:
            break
        for c in line:
            if c == "{":
                if open_p == 0:
                    line_start = i
                open_p += 1
            if c == "}":
                open_p -= 1
                if open_p == 0:
                    line_end = i
                    z = True
    return line_start, line_end + 1

def find_class_attrib(string, attrib_re):
    open_p = 0
    list_attrib = []
    for line in string:
        for c in line:
            if c == "{":
                open_p += 1
            if c == "}":
                open_p -= 1
        if open_p == 1:
            list_attrib.extend([i.group() for i in re.finditer(attrib_re, line)])
    return list_attrib

def remove_cmt(string, cmt_re):
    # print(string)
    list_match = [i for i in re.finditer(cmt_re, string)]
    for i in list_match:
        start = i.start()
        end = i.end()
        new_string = ""
        new_string += string[:start]
        new_string += re.sub("[^\n]", " ", string[start:end])
        new_string += string[end:]
        string = new_string
    return string

def find_method_line(lines, start_class, start_bug, end_class, method_re):
    string = lines[start_class - 1:start_bug]
    # print("\n".join(string))
    method_match = [i for i in re.finditer(method_re, "\n".join(string), re.M)]
    # print(method_match)
    if len(method_match) <= 0:
        return 0, 0
    method = method_match[-1].group()
    method_start = string.index(method.split("\n")[0])
    # print("\n".join(lines[method_start + start_class - 1: end_class + 1]))
    line_start, line_end = find_parentless_line(lines[method_start + start_class - 1: end_class + 1])
    return method_start + start_class - 1, line_end + method_start + start_class - 1


def test_parser():
    path = "D://study//thesis//data//code//defect_detect//org//owasp//benchmark//testcode//BenchmarkTest00059.java"
    with open(path) as file:
        data = file.read().strip()
        print(data)
        tree = parse(data)
        for p, node in tree:
            print(node)


if __name__ == "__main__":
    bug_instance = """<BugInstance rank="15" category="SECURITY" instanceHash="98497b1a447356bffc77e14342efb7dc" instanceOccurrenceNum="0" priority="3" abbrev="SECCU" type="COOKIE_USAGE" instanceOccurrenceMax="2">
    <ShortMessage>Potentially Sensitive Data in Cookie</ShortMessage>
    <LongMessage>Sensitive data may be stored by the application in a cookie.</LongMessage>
    <Class classname="org.owasp.benchmark.testcode.BenchmarkTest02000" primary="true">
        <SourceLine start="30" classname="org.owasp.benchmark.testcode.BenchmarkTest02000" sourcepath="org/owasp/benchmark/testcode/BenchmarkTest02000.java" sourcefile="BenchmarkTest02000.java" end="105">
            <Message>At BenchmarkTest02000.java:[lines 30-105]</Message>
        </SourceLine>
        <Message>In class org.owasp.benchmark.testcode.BenchmarkTest02000</Message>
    </Class>
    <Method isStatic="false" classname="org.owasp.benchmark.testcode.BenchmarkTest02000" name="doPost" primary="true" signature="(Ljavax/servlet/http/HttpServletRequest;Ljavax/servlet/http/HttpServletResponse;)V">
        <SourceLine endBytecode="1021" startBytecode="0" start="41" classname="org.owasp.benchmark.testcode.BenchmarkTest02000" sourcepath="org/owasp/benchmark/testcode/BenchmarkTest02000.java" sourcefile="BenchmarkTest02000.java" end="99"></SourceLine>
        <Message>In method org.owasp.benchmark.testcode.BenchmarkTest02000.doPost(HttpServletRequest, HttpServletResponse)</Message>
    </Method>
    <SourceLine endBytecode="272" startBytecode="272" start="78" classname="org.owasp.benchmark.testcode.BenchmarkTest02000" primary="true" sourcepath="org/owasp/benchmark/testcode/BenchmarkTest02000.java" sourcefile="BenchmarkTest02000.java" end="78">
        <Message>At BenchmarkTest02000.java:[line 78]</Message>
    </SourceLine>
    </BugInstance>"""
    # preproces('../elastic/data0/ElasticSearchException.java', [], ".")
    # print()
    with open("../vocab") as file:
        vocab = [line.strip() for line in file.readlines()]
    # print(vocab)
    preprocess_ver2("../spotbugsXml.xml", vocab)
    # test_parser()