import pandas as pd
import os
from shutil import copy
from javalang.parse import parse
from javalang.parser import JavaSyntaxError
from javalang.tokenizer import tokenize
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import JsonExporter
import re
import xml.etree.ElementTree as ET

SPECIAL = ["<UNK>", "<BOF>", "<EOF>"]
DATA_PATH = "../"


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
    tree = ET.parse(xml_file)
    root = tree.getroot()
    print("Extracting xml file!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    previous = ""
    bug_instances = []
    for ele in root:
        if ele.tag == "BugInstance":
            for child in ele:
                if child.tag == "SourceLine":
                    previous = child.attrib['sourcepath']
                    break
            break
    for ele in root:
        if ele.tag == "BugInstance":
            for child in ele:
                if child.tag == "SourceLine":
                    attrib = child.attrib
                    if previous == attrib['sourcepath']:
                        bug_instances.append(attrib)
                    else:
                        extract_data(previous, bug_instances, vocab)
                        bug_instances = [attrib]
                        previous = attrib['sourcepath']
    extract_data(previous, bug_instances, vocab)


def extract_data(filename, bug_instances, vocab):
    list_re = []
    # print("Extracting " + filename + " tree!!!!!!!!!!!!!!!!!!!")
    match_cmt = ""
    match_class = ""
    with open(os.path.join(DATA_PATH, filename), encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
        for bug in bug_instances:
            start = bug['start']
            end = bug['end']
            for line in range(int(start), int(end) + 1):
                clean_cmt = re.sub(r"", " ", lines[line])
                clean = re.sub(r"[\"\'\s;(){}=+\-/!~<>.,%^&*#$|]+", " ", lines[line])
                list_clean = [i for i in clean.split(" ") if i != ""]
                if len(list_clean) > 0:
                    list_re.append("|".join(list_clean))
    with open(os.path.join(DATA_PATH, filename), encoding='utf-8') as file:
        try:
            node = parse(file.read().strip())
            start = Node("<BOF>")
            traveler(node, start, list_re)
            defect_node = ['' for i in range(len(list_re))]
            max_value = [0 for i in range(len(list_re))]
            for n in PreOrderIter(start):
                for i in range(len(n.len_match)):
                    if n.len_match[i] > max_value[i]:
                        max_value[i] = n.len_match[i]
                        defect_node[i] = n
                        delattr(n, "len_match")
            for n in defect_node:
                n.isBug = True

            for pre, fill, n in RenderTree(start):
                vocab.append(n.name)
        except Exception or AttributeError:
            print(filename)
            return ''
    exporter = JsonExporter()
    return exporter.export(start)


def traveler(parse_node, tree_node, list_regex):
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
    new_node.len_match = [len(re.findall(i, parse_node.__str__())) for i in list_regex]
    list_child = []
    for i, n in enumerate(parse_node.children):
        if isinstance(n, list):
            for instance in n:
                if re.match(r"\<class \'javalang\.tree\..*\'\>", str(type(instance))):
                    list_child.append(instance)
        if re.match(r"\<class \'javalang\.tree\..*\'\>", str(type(n))):
            list_child.append(n)
    if len(list_child) == 0:
        return
    else:
        # print(list_child)
        for child in list_child:
            traveler(child, new_node, list_regex)

def search_and_replace(pattern, string):
    pass

def get_branch():
    pass


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
    print(preprocess_ver2("../test.xml", []))