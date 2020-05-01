from utils.utils import preproces
import plyj.parser as plyj

if __name__=="__main__":
    # preproces()
    parser = plyj.Parser()
    test = """
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
    """
    tree = parser.parse_expression(test)
    print(tree)
