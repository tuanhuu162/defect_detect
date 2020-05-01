import os
from javalang.parse import parse
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import JsonExporter
from copy import deepcopy
import re
from lxml import etree
import json

OUTDIR = "./json_data/"
DATA_PATH = "./src"

def process_juliet(xml_file, vocab):

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
            isBug = 0
            # class_attrib = []
            virtual_data = {
                "class": [],
                "field": [],
                "method": []
            }
            for child in ele:
                # class_name = ""
                if child.tag == "Class":
                    if ("testcases." in child.attrib['classname'] or "testcasesupport." in child.attrib['classname']) and 'role' not in child.attrib:
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
                    if "testcases." in child.attrib['classname']:
                        if len(re.findall("bad", child.attrib['name'])) > 0:
                            isBug = 1  # true positive
                        if child.tag == "Method":
                            for c in child:
                                if c.tag == "SourceLine":
                                    if ele.attrib['type'] in ["UC_USELESS_VOID_METHOD", "ESync_EMPTY_SYNC"]:
                                        bug_instances.append(c.attrib)
                                    # print(c.attrib)
                                    if 'start' in c.attrib:
                                        virtual_data["method"].append((int(c.attrib['start']), int(c.attrib['end'])))
                    else:
                        virtual_data["field"].append(child.attrib['name'])
                if child.tag == "SourceLine":
                    if ele.attrib['type'] not in ["UC_USELESS_VOID_METHOD", "ESync_EMPTY_SYNC"]:
                        attrib = child.attrib
                        bug_instances.append(attrib)
            if len(bug_instances) > 0:
                # extract_data(class_attrib, bug_instances, vocab, list_data, isBug)
                extract_data(virtual_data, bug_instances, vocab, list_data, isBug)
            else:
                print("DO NOT HANDLE CLASS ERROR")
    vocab = set(vocab)
    with open(os.path.join(OUTDIR, "vocab"), 'w') as file:
        for i in vocab:
            file.write(i + "\n")
    with open(os.path.join(OUTDIR, "data_juliet.json"), 'w') as file:
        file.write(json.dumps(list_data))

# def extract_data(filename, bug_instances, vocab, data_file, list_label):
# def extract_data(class_attrib, bug_instances, vocab, list_data, label):
#     base_name = class_attrib[0].split("/")[-1].split(".")[0]
def extract_data(virtual_data, bug_instances, vocab, list_data, label):
    print(virtual_data)
    base_name = virtual_data['class'][0].split("/")[-1].split(".")[0]
    match_cmt = r"\/\/.*"
    match_class = "(\w+\.)+[A-Z]\w+"
    match_string = r"\"([^\"]+)\"|\'([^\"']+)\'"
    exporter = JsonExporter()
    start = Node("<BOF>")
    print("Extracting " + base_name + " tree!!!!!!!!!!!!!!!!!!!")
    # print(bug_instances)
    # virtual_file, list_bug = make_virtual_class(class_attrib, bug_instances)
    try:
        virtual_file, list_bug = make_virtual_class(virtual_data, bug_instances)
        node = parse(virtual_file)
    except Exception:
        print(base_name)
        return ''
    lines = virtual_file.split("\n")
    # print(virtual_file)
    list_re = []
    for bug in list_bug:
        start_line = int(bug[0])
        end = int(bug[1])
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
    virtual_class = ""
    cmt_re = r"/\*([^*]|[\r\n]|(\*+([^*/]|[\r\n])))*\*+/"
    import_re = r"(import|package).*\;"
    class_re = r"(public|private|protected|default)?\s{1,3}class\s{1,3}[\w\s\n\-\.\<\>\[\]\,]*\{"
    method_re = r"(((public|private|protected|default)\s{1,3}(static\s{1,3})?)|(static\s{1,3}(public|private|protected|default)\s{1,3}))?(\w+[\w\s\.\<\>\,\[\]\-]+)\s{1,3}\w[\w-]+\s{0,3}\([\w\s\.\<\>\,\[\]\-]*\)[\w\s\.\<\>\,\[\]\-]*\{"
    # attrib_re = "(((public|private|protected|default)\s{1,3}(static\s{1,3})?)|((static\s{1,3})(public|private|protected|default)\s{1,3}))?\w[\w\.\-<>\*\[\]\s]+\s{1,3}\w[\w-]+(\s{0,3}=.*\n{0,3}.*)?;"
    attrib_re = "((public|private|protected|default|static|final){1,3}\s{1,3})?\w[\w\.\-<>\*\[\]\s]+\s{1,3}\w[\w\.\-<>\*\[\]]+(\s{0,3}=.*\n{0,3}.*)?;"
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
    print(virtual_class)
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
    method_match = [i for i in re.finditer(method_re, "\n".join(string))]
    if len(method_match) <= 0:
        return 0, 0
    method = method_match[-1].group()
    method_start = string.index(method.split("\n")[0])
    # print("\n".join(lines[method_start + start_class - 1: end_class + 1]))
    line_start, line_end = find_parentless_line(lines[method_start + start_class - 1: end_class + 1])
    return method_start + start_class - 1, line_end + method_start + start_class - 1


def check_parser():
    path = "../BenchmarkTest07216.java"
    with open(path) as file:
        data = file.read().strip()
        print(data)
        tree = parse(data)

if __name__=="__main__":
    # preproces('../elastic/data0/ElasticSearchException.java', [], ".")
    # print()
    # with open(os.path.join(DATA_PATH, "testcases/CWE113_HTTP_Response_Splitting/s01/CWE113_HTTP_Response_Splitting__Environment_addCookieServlet_02.java")) as file:
    #     string = file.read()
    # # print(len(remove_cmt(string, "\/\*.*\s*.*\*\/").split("\n")), len(string.split("\n")))
    # print(remove_cmt(string, "\/\*.*\s*.*\*\/"))
    with open("./json_data/vocab") as file:
        vocab = [line.strip() for line in file.readlines()]
    process_juliet("./spotbug2.xml", vocab)
    # test_parser()
    # print(make_virtual_class(["testcases/CWE113_HTTP_Response_Splitting/s01/CWE113_HTTP_Response_Splitting__Environment_addCookieServlet_22a.java", 23, 101], [{"start": "27", "end": "27"}]))
    # print(make_virtual_class(["testcases/CWE113_HTTP_Response_Splitting/s01/CWE113_HTTP_Response_Splitting__Environment_addCookieServlet_12.java", 25, 163], [{"start": "86", "end": "86"}]))
    # print(make_virtual_class(["testcases/CWE113_HTTP_Response_Splitting/s01/CWE113_HTTP_Response_Splitting__Environment_addCookieServlet_21.java", 25, 160], [{"start": "38", "end": "38"}, {"start": "44", "end": "44"}]))