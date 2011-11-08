/***
 * @file save_restore_model_test.cpp
 * @author Neil Slagle
 *
 * Here we have tests for the SaveRestoreModel class.
 */

#include <mlpack/core/utilities/save_restore_utility.hpp>

#include <boost/test/unit_test.hpp>
#define ARGSTR(a) a,#a

BOOST_AUTO_TEST_SUITE(SaveRestoreUtilityTests);

  /***
   * Exhibit proper save restore utility usage
   *  of child class proper usage
   */
  class SaveRestoreTest
  {
    private:
      size_t anInt;
      SaveRestoreUtility saveRestore;
    public:
      SaveRestoreTest()
      {
        saveRestore = SaveRestoreUtility();
      }
      bool SaveModel (std::string filename)
      {
        saveRestore.SaveParameter (anInt, "anInt");
        return saveRestore.WriteFile (filename);
      }
      bool LoadModel (std::string filename)
      {
        bool success = saveRestore.ReadFile (filename);
        if (success)
        {
          anInt = saveRestore.LoadParameter (anInt, "anInt");
        }
        return success;
      }
      const size_t AnInt () { return anInt; }
      void AnInt (size_t s) { this->anInt = s; }
  };

  /***
   * Perform a save and restore on basic types.
   */
  BOOST_AUTO_TEST_CASE(save_basic_types)
  {
    bool b = false;
    char c = 67;
    unsigned u = 34;
    size_t s = 12;
    short sh = 100;
    int i = -23;
    float f = -2.34f;
    double d = 3.14159;
    std::string cc = "Hello world!";

    SaveRestoreUtility* sRM = new SaveRestoreUtility();

    sRM->SaveParameter (ARGSTR(b));
    sRM->SaveParameter (ARGSTR(c));
    sRM->SaveParameter (ARGSTR(u));
    sRM->SaveParameter (ARGSTR(s));
    sRM->SaveParameter (ARGSTR(sh));
    sRM->SaveParameter (ARGSTR(i));
    sRM->SaveParameter (ARGSTR(f));
    sRM->SaveParameter (ARGSTR(d));
    sRM->SaveParameter (ARGSTR(cc));
    sRM->WriteFile ("test_basic_types.xml");

    sRM->ReadFile ("test_basic_types.xml");

    bool b2 =         sRM->LoadParameter (ARGSTR(b));
    char c2 =         sRM->LoadParameter (ARGSTR(c));
    unsigned u2 =     sRM->LoadParameter (ARGSTR(u));
    size_t s2 =       sRM->LoadParameter (ARGSTR(s));
    short sh2 =       sRM->LoadParameter (ARGSTR(sh));
    int i2 =          sRM->LoadParameter (ARGSTR(i));
    float f2 =        sRM->LoadParameter (ARGSTR(f));
    double d2 =       sRM->LoadParameter (ARGSTR(d));
    std::string cc2 = sRM->LoadParameter (ARGSTR(cc));

    BOOST_REQUIRE (b == b2);
    BOOST_REQUIRE (c == c2);
    BOOST_REQUIRE (u == u2);
    BOOST_REQUIRE (s == s2);
    BOOST_REQUIRE (sh == sh2);
    BOOST_REQUIRE (i == i2);
    BOOST_REQUIRE (cc == cc2);
    BOOST_REQUIRE_CLOSE (f, f2, 1e-5);
    BOOST_REQUIRE_CLOSE (d, d2, 1e-5);

    delete sRM;
  }

  /***
   * Test the arma::mat functionality.
   */
  BOOST_AUTO_TEST_CASE(save_arma_mat)
  {
    arma::mat matrix;
    matrix <<  1.2 << 2.3 << -0.1 << arma::endr
           <<  3.5 << 2.4 << -1.2 << arma::endr
           << -0.1 << 3.4 << -7.8 << arma::endr;
    SaveRestoreUtility* sRM = new SaveRestoreUtility();

    sRM->SaveParameter (ARGSTR (matrix));

    sRM->WriteFile ("test_arma_mat_type.xml");

    sRM->ReadFile ("test_arma_mat_type.xml");

    arma::mat matrix2 = sRM->LoadParameter (ARGSTR (matrix));

    for (size_t row = 0; row < matrix.n_rows; ++row)
    {
      for (size_t column = 0; column < matrix.n_cols; ++column)
      {
        BOOST_REQUIRE_CLOSE(matrix(row,column), matrix2(row,column), 1e-5);
      }
    }

    delete sRM;
  }
  /***
   * Test SaveRestoreModel proper usage in child classes and loading from
   *   separately defined objects
   */
  BOOST_AUTO_TEST_CASE(save_restore_model_child_class_usage)
  {
    SaveRestoreTest* saver = new SaveRestoreTest();
    SaveRestoreTest* loader = new SaveRestoreTest();
    size_t s = 1200;
    const char* filename = "anInt.xml";

    saver->AnInt (s);
    saver->SaveModel (filename);
    delete saver;

    loader->LoadModel (filename);

    BOOST_REQUIRE (loader->AnInt () == s);

    delete loader;
  }

BOOST_AUTO_TEST_SUITE_END();