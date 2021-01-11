""" Strip comments and docstrings from a file.
"""
import re


def remove_python_style_comments(text):
    new_text = text
    regex = re.compile(r"((\/\*)|(\"\"\"))[\s\S]*?((\*\/)|(\"\"\"))|([^:]|^)((\/\/)|#).*?$", re.MULTILINE | re.DOTALL)

    while True:
        old = new_text
        new_text = regex.sub("", new_text)

        if old == new_text:
            break
    return new_text


data = """
RegExr was created by gskinner.com, and is proudly hosted by Media Temple.
// Kiki

/*

*/

\"\"\"
dscd
\"\"\"

// miki
Edit the Expression"&Text to see matches. Roll over matches or the expression for details. PCRE & JavaScript flavors of RegEx are supported. Validate your expression with Tests mode.

The side bar includes a Cheatsheet, full Reference, and Help. You can also Save & Share with the Community, and view patterns you create or favorite in My Patterns.

Explore results with the Tools below. Replace & List output custom results. Details lists capture groups. Explain describes your expression in plain English.

import java.io.*;

/**
* <h1>Add Two Numbers!</h1>
* The AddNum program implements an application that
* simply adds two given integer numbers and Prints
* the output on the screen.
* <p>
* <b>Note:</b> Giving proper comments in your program makes it more
* user friendly and it is assumed as a high quality code.
*
* @author  Zara Ali
* @version 1.0
* @since   2014-03-31
*/
public class AddNum {
   /**
   * This method is used to add two integers. This is
   * a the simplest form of a class method, just to
   * show the usage of various javadoc Tags.
   * @param numA This is the first paramter to addNum method
   * @param numB  This is the second parameter to addNum method
   * @return int This returns sum of numA and numB.
   */
   public int addNum(int numA, int numB) {
      return numA + numB;
   }

   /**
   * This is the main method which makes use of addNum method.
   * @param args Unused.
   * @return Nothing.
   * @exception IOException On input error.
   * @see IOException
   */

   public static void main(String args[]) throws IOException {
      AddNum obj = new AddNum();
      int sum = obj.addNum(10, 20);

      System.out.println("Sum of 10 and 20 is :" + sum);
   }
}

#kiki
#siki
\"\"\"
Komentar
\"\"\"
print(5)

"""
print(data)
print("=====================================")
print(f'REMOVED: "{remove_python_style_comments(data)}"')
