package ch7;

import java.util.Arrays;

class MyClass{
  public void printFirstElement(int[] a) {
    System.out.println("The first element is " + a[0]);
  }
  public int[] returnArray() {
    int[] array = new int[3];
    for(int i = 0; i < array.length; ++i) {
      array[i] = i * 2;
    }
    return array;
  }
}

public class ArrayMethodDemo {
  public static void main(String[] args) {
    MyClass amd = new MyClass();

    int[] myArray = {1,2,3,4,5};
    amd.printFirstElement(myArray);

    int[] myArray2 = amd.returnArray();
    System.out.println(Arrays.toString(myArray2));
  }
}
