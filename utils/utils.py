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
package org.owasp.benchmark.testcode;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet("/BenchmarkTest00001")
public class BenchmarkTest00001 extends HttpServlet {

	private static final long serialVersionUID = 1L;

	@Override
	public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		doPost(request, response);
	}

	@Override
	public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// some code

		javax.servlet.http.Cookie[] cookies = request.getCookies();

		String param = null;
		boolean foundit = false;
		if (cookies != null) {
			for (javax.servlet.http.Cookie cookie : cookies) {
				if (cookie.getName().equals("foo")) {
					param = cookie.getValue();
					foundit = true;
				}
			}
			if (!foundit) {
				// no cookie found in collection
				param = "";
			}
		} else {
			// no cookies
			param = "";
		}


		java.security.Provider[] provider = java.security.Security.getProviders();
		javax.crypto.Cipher c;

		try {
			if (provider.length > 1) {
				c = javax.crypto.Cipher.getInstance("DES/CBC/PKCS5PADDING", java.security.Security.getProvider("SunJCE"));
			} else {
				c = javax.crypto.Cipher.getInstance("DES/CBC/PKCS5PADDING", java.security.Security.getProvider("SunJCE"));
			}
		} catch (java.security.NoSuchAlgorithmException e) {
			System.out.println("Problem executing crypto - javax.crypto.Cipher.getInstance(java.lang.String,java.security.Provider) Test Case");
			throw new ServletException(e);
		} catch (javax.crypto.NoSuchPaddingException e) {
			System.out.println("Problem executing crypto - javax.crypto.Cipher.getInstance(java.lang.String,java.security.Provider) Test Case");
			throw new ServletException(e);
		}
		response.getWriter().println("Crypto Test javax.crypto.Cipher.getInstance(java.lang.String,java.security.Provider) executed");
	}
}

"""
    with open(filename, encoding='utf-8') as file:
        try:
            node = parse(file.read().strip())
            start = Node("<BOF>")
            traveler(node, start)
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
    traveler(node_1, start_1)
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
        if data.loc[i, ' real vulnerability'] == " true" and data.loc[i, ' pass/fail'] == " pass":
            list_label[data.loc[i, '# test name']] = 1
        elif data.loc[i, ' real vulnerability'] == " true" and data.loc[i, ' pass/fail'] == " fail":
            list_label[data.loc[i, '# test name']] = 0
        else:
            list_label[data.loc[i, '# test name']] = 2


    previous = ""
    bug_instances = []
    for ele in tree:
        if ele.tag == "BugInstance":
            for child in ele:
                if child.tag == "SourceLine":
                    previous = child.attrib['sourcepath']
                    break
            break

    for ele in tree:
        if ele.tag == "BugInstance":
            for child in ele:
                if child.tag == "SourceLine":
                    attrib = child.attrib
                    if previous == attrib['sourcepath']:
                        bug_instances.append(attrib)
                    else:
                        extract_data(previous, bug_instances, vocab, list_label)
                        bug_instances = [attrib]
                        previous = attrib['sourcepath']
    extract_data(previous, bug_instances, vocab, list_label)

    vocab = set(vocab)
    with open(os.path.join(OUTDIR, "vocab"), 'w') as file:
        for i in vocab:
            file.write(i + "\n")


def extract_data(filename, bug_instances, vocab, list_label):
    list_re = []
    base_name = filename.split("/")[-1].split(".")[0]
    if not re.match(r"BenchmarkTest.*", base_name):
        return
    if list_label[base_name] == 2:
        return
    print("Extracting " + base_name + " tree!!!!!!!!!!!!!!!!!!!")
    match_cmt = r"\/\/.*"
    match_class = "(\w+\.)+[A-Z]\w+"
    match_string = r"\"([^\"]+)\"|\'([^\"']+)\'"
    with open(os.path.join(DATA_PATH, filename), encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
        for bug in bug_instances:
            start = bug['start']
            end = bug['end']
            for line in range(int(start) - 1, int(end)):
                clean_cmt = re.sub(match_cmt, " ", lines[line])
                clean_string, list_string = search_and_replace(match_string, clean_cmt)
                clean_class, list_class = search_and_replace(match_class, clean_string)
                clean = re.sub(r"[\s;\]\[(){}=+\-/?:!~<>.,%^&*#$|]+", " ", clean_class)
                list_clean = [i for i in clean.split(" ") if i != ""]
                list_clean.extend([i.group().replace("\"", r"\"") for i in list_string])
                list_clean.extend([i.group().replace(".", r"\.") for i in list_class])
                list_clean.extend([i.group().replace(".", r"\.") for i in list_class])
                if len(list_clean) > 0:
                    list_re.append("|".join(list_clean))
    with open(os.path.join(DATA_PATH, filename), encoding='utf-8') as file:
        try:
            node = parse(file.read().strip())
            start = Node("<BOF>")
            len_match = [0 for i in range(len(list_re))]
            traveler(node, start, list_re, len_match)
            del node
            defect_node = ['' for i in range(len(list_re))]
            max_value = [0 for i in range(len(list_re))]
            for n in PreOrderIter(start):
                if hasattr(n, "len_match"):
                    for i in range(len(n.len_match)):
                        if n.len_match[i] > max_value[i]:
                            max_value[i] = n.len_match[i]
                            defect_node[i] = n
            for i, n in enumerate(defect_node):
                n.regex = list_re[i]
                n.isBug = True

            for pre, fill, n in RenderTree(start):
                vocab.append(n.name)
        except IOError:
            print(filename)
            return ''
    exporter = JsonExporter()
    list_bug, list_method = get_branch(start, defect_node)
    
    list_data = {
        "bug": [],
        "method": [],
        "label": []
    }
    for i in range(len(list_bug)):
        list_data["bug"].append(exporter.export(list_bug[i]))
        list_data["method"].append(exporter.export(list_method[i]))
        list_data["label"].append(list_label[base_name])
    with open(os.path.join(OUTDIR, "data.json"), "w") as file:
        file.write(json.dumps(list_data))
    print("Finish !!!!!!!!!!!!!!!!!!!")


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
        new_node.attr = [(attr, getattr(parse_node, attr)) for attr in parse_node.attrs if isinstance(getattr(parse_node, attr), str)]
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

def get_branch(start_node, defect_node):
    walker = Walker()
    list_bug = []
    list_method = []
    exporter = JsonExporter()
    for node in defect_node:
        up_node, current, downnode = walker.walk(start_node, node)
        start = Node(current.name)
        list_bug.append(start)
        start_method = Node(current.name)
        list_method.append(start_method)
        for n in downnode:
            if n.name == "MethodDeclaration":
                new_node = deepcopy(n)
                new_node.parent = start_method
                break
            new_node = Node(n.name, parent=start_method)
            start_method = new_node
        for n in downnode:
            new_node = Node(n.name, parent=start)
            start = new_node
    # for i in range(len(list_bug)):
    #     print(exporter.export(list_bug[i]))
    #     print(exporter.export(list_method[i]))

    return list_bug, list_method

def test_parser():
    path = "../org/owasp/benchmark/testcode/BenchmarkTest06923.java"
    with open(path) as file:
        data = file.read().strip()
        print(data)
        tree = parse(data)


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
    preprocess_ver2("../spotbugsXml.xml", [])
    # test_parser()