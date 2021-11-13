package ch9;

import java.util.ArrayList;

public class ArrayListDemo {
  public static void main(String args[]) {
    ArrayList<Integer> arrayList = new ArrayList<>();
    arrayList.add(40);
    arrayList.add(53);
    arrayList.add(45);
    arrayList.add(53);

    arrayList.add(2, 51);
    System.out.println(arrayList);

    arrayList.set(3, 49);
    System.out.println(arrayList);

    arrayList.remove(3);
    System.out.println(arrayList);

    System.out.println(arrayList.get(2));

    System.out.println(arrayList.size());

  }
}