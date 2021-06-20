'''

Description
Implement the class ReferenceManager. Include the following two methods:

copyValue(Node obj). This would just copy the value from parameter obj to
the public attribute node. But obj and node are still two difference instances / objects.
copyReference(Node obj). This would copy the reference from obj to node.
So that both node and obj are point to the same object.


java写法

public class ReferenceManager {
    public Node node;

    public void copyValue(Node obj) {
        if (obj == null) {
            return;
        }
        if (node == null) {
            node = new Node(obj.val);
        }
        node.val = obj.val;
    }

    public void copyReference(Node obj) {
        node = obj;
    }
}



'''