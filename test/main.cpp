#include "easypr.h"
#include "easypr/util/switch.hpp"

#include "accuracy.hpp"
#include "chars.hpp"
#include "plate.hpp"
#include "easypr/core/core_func.h"

// %OPENCV%\x86\vc12\lib opencv_world300d.lib;

namespace easypr {

    namespace demo {

        // interactions

        int accuracyTestMain() {
            std::shared_ptr<easypr::Kv> kv(new easypr::Kv);
            kv->load("resources/text/chinese_mapping");

            bool isExit = false;
            while (!isExit) {
                easypr::Utils::print_file_lines("resources/text/batch_test_menu");
                std::cout << kv->get("make_a_choice") << ":";

                int select = -1;
                bool isRepeat = true;
                Result result;

                while (isRepeat) {
                    std::cin >> select;
                    isRepeat = false;
                    switch (select) {
                        case 1:
                            accuracyTest("resources/image/general_test", result);
                            break;
                        case 2:
                            accuracyTest("resources/image/native_test", result);
                            break;
                        case 3:
                            gridSearchTest("resources/image/general_test");
                            break;
                        case 4:
                            isExit = true;
                            break;
                        default:
                            std::cout << kv->get("input_error") << ":";
                            isRepeat = true;
                            break;
                    }
                }
            }
            return 0;
        }

        int trainChineseMain() {
            std::shared_ptr<easypr::Kv> kv(new easypr::Kv);
            kv->load("resources/text/chinese_mapping");

            bool isExit = false;
            while (!isExit) {
                easypr::Utils::print_file_lines("resources/text/train_menu");
                std::cout << kv->get("make_a_choice") << ":";

                int select = -1;
                bool isRepeat = true;
                while (isRepeat) {
                    std::cin >> select;
                    isRepeat = false;
                    switch (select) {
                        case 1: {
                            easypr::AnnChTrain ann("tmp/annCh", "tmp/annCh.xml");
                            ann.setNumberForCount(100);
                            ann.train();
                        }
                            break;
                        case 2: {
                            easypr::AnnChTrain ann("tmp/annCh", "tmp/annCh.xml");
                            ann.setNumberForCount(350);
                            ann.train();
                        }
                            break;
                        case 3: {
                            easypr::AnnChTrain ann("tmp/annCh", "tmp/annCh.xml");
                            ann.setNumberForCount(700);
                            ann.train();
                        }
                            break;
                        case 4: {
                            easypr::AnnChTrain ann("tmp/annCh", "tmp/annCh.xml");
                            ann.setNumberForCount(1000);
                            ann.train();
                        }
                            break;
                        case 5: {
                            easypr::AnnChTrain ann("tmp/annCh", "tmp/annCh.xml");
                            ann.setNumberForCount(1500);
                            ann.train();
                        }
                            break;
                        case 6:
                            isExit = true;
                            break;
                        default:
                            std::cout << kv->get("input_error") << ":";
                            isRepeat = true;
                            break;
                    }
                }
            }
            return 0;
        }

        int testMain() {
            std::shared_ptr<easypr::Kv> kv(new easypr::Kv);
            kv->load("resources/text/chinese_mapping");

            bool isExit = false;
            while (!isExit) {
                Utils::print_file_lines("resources/text/test_menu");
                std::cout << kv->get("make_a_choice") << ":";

                int select = -1;
                bool isRepeat = true;
                while (isRepeat) {
                    std::cin >> select;
                    isRepeat = false;
                    switch (select) {
                        case 1:
                            assert(test_plate_locate() == 0);
                            break;
                        case 2:
                            assert(test_plate_judge() == 0);
                            break;
                        case 3:
                            assert(test_plate_detect() == 0);
                            break;
                        case 4:
                            assert(test_chars_segment() == 0);
                            break;
                        case 5:
                            assert(test_chars_identify() == 0);
                            break;
                        case 6:
                            assert(test_chars_recognise() == 0);
                            break;
                        case 7:
                            assert(test_plate_recognize() == 0);
                            break;
                        case 8:
                            assert(test_plate_locate() == 0);
                            assert(test_plate_judge() == 0);
                            assert(test_plate_detect() == 0);

                            assert(test_chars_segment() == 0);
                            assert(test_chars_identify() == 0);
                            assert(test_chars_recognise() == 0);

                            assert(test_plate_recognize() == 0);
                            break;

                        default:
                            break;
                    }
                }
            }
            return 0;
        }

    }  // namespace demo

}  // namespace easypr

void command_line_handler(int argc, const char *argv[]) {
    program_options::Generator options;

    options.add_subroutine("svm", "svm operations").make_usage("Usage:");
    {
        /* ------------------------------------------
         | SVM Training operations
         | ------------------------------------------
         |
         | $ demo svm --plates=path/to/plates/ [--test] --svm=save/to/svm.xml
         |
         | ------------------------------------------
         */
        options("h,help", "show help information");
        options(",plates", "",
                "a folder contains both forward data and inverse data in the "
                "separated subfolders");
        options(",svm", easypr::kDefaultSvmPath, "the svm model file");
        options("t,test", "run tests in --plates");
    }

    options.add_subroutine("ann", "ann operation").make_usage("Usages:");
    {
        /* ------------------------------------------
        | ANN_MLP Training operations
        | ------------------------------------------
        |
        | $ demo ann --zh-chars=zhchars/ --en-chars=enchars/ --ann=save/to/ann.xml
        |
        | ------------------------------------------
        */
        options("h,help", "show help information");
        options(",chars", "",
                "the folder contains character sub-folders, with each folder"
                "named by label defined in include/easypr/config.h");
        options(",ann", easypr::kDefaultAnnPath,
                "the ann model file you want to save");
        options("t,test", "run test in --chars");
    }

    options.add_subroutine("locate", "locate plates in an image")
            .make_usage("Usage:");
    {
        /* ------------------------------------------
        | Plate locating operations
        | ------------------------------------------
        |
        | $ demo locate -f file
        |
        | ------------------------------------------
        */
        options("h,help", "show help information");
        options("f,file", "",
                "the target picture which contains one or more plates");
    }

    options.add_subroutine(
                    "judge", "determine whether an image block is the license plate")
            .make_usage("Usage:");
    {
        /* ------------------------------------------
        | Plate judge operations
        | ------------------------------------------
        |
        | $ demo judge -f file --svm model/svm.xml
        |
        | ------------------------------------------
        */
        options("h,help", "show help information");
        options("f,file", "the target image block");
        options(",svm", easypr::kDefaultSvmPath, "the svm model file");
    }

    options.add_subroutine("recognize", "plate recognition").make_usage("Usage:");
    {
        /* ------------------------------------------
        | Plate recognize operations
        | ------------------------------------------
        |
        | $ demo recognize -p file --svm model/svm.xml
        |                          --ann model/ann.xml
        | $ demo recognize -pb dir/ --svm model/svm.xml
        |                           --ann model/ann.xml
        |
        | ------------------------------------------
        */
        options("h,help", "show help information");
        options("p,path", "", "where is the target picture or target folder");
        options("b,batch", "do batch recognition, if set, --path means a folder");
        options("c,color", "returns the plate color, blue or yellow");
        options(",svm", easypr::kDefaultSvmPath, "the svm model file");
        options(",ann", easypr::kDefaultAnnPath, "the ann model file");
    }

    auto parser = options.make_parser();

    try {
        parser->parse(argc, argv);
    } catch (const std::exception &err) {
        std::cout << err.what() << std::endl;
        return;
    }

    auto subname = parser->get_subroutine_name();

    program_options::select(subname)
            .found("svm",
                   [&]() {
                       if (parser->has("help") || argc <= 2) {
                           std::cout << options("svm");
                           return;
                       }

                       easypr::SvmTrain svm(parser->get("plates")->c_str(),
                                            parser->get("svm")->c_str());

                       if (parser->has("test")) {
                           svm.test();
                       } else {
                           svm.train();
                       }
                   })
            .found("ann",
                   [&]() {
                       if (parser->has("help") || argc <= 2) {
                           std::cout << options("ann");
                           return;
                       }

                       assert(parser->has("chars"));
                       assert(parser->has("ann"));

                       easypr::AnnTrain ann(parser->get("chars")->c_str(),
                                            parser->get("ann")->c_str());

                       if (parser->has("test")) {
                           ann.test();
                       } else {
                           ann.train();
                       }
                   })
            .found("locate",
                   [&]() {
                       if (parser->has("help") || argc <= 2) {
                           std::cout << options("locate");
                           return;
                       }

                       if (parser->has("file")) {
                           easypr::api::plate_locate(parser->get("file")->val().c_str());
                           std::cout << "finished, results can be found in tmp/"
                                     << std::endl;
                       }
                   })
            .found("judge",
                   [&]() {
                       if (parser->has("help") || argc <= 2) {
                           std::cout << options("judge");
                           std::cout << "Note that the input image's size should "
                                     << "be the same as the one you gived to svm train."
                                     << std::endl;
                           return;
                       }

                       if (parser->has("file")) {
                           assert(parser->has("file"));
                           assert(parser->has("svm"));

                           auto image = parser->get("file")->val();
                           auto svm = parser->get("svm")->val();

                           const char *true_or_false[2] = {"false", "true"};

                           std::cout << true_or_false[easypr::api::plate_judge(
                                   image.c_str(), svm.c_str())]
                                     << std::endl;
                       }
                   })
            .found("recognize",
                   [&]() {
                       if (parser->has("help") || argc <= 2) {
                           std::cout << options("recognize");
                           return;
                       }

                       if (parser->has("path")) {
                           if (parser->has("batch")) {
                               // batch testing
                               auto folder = parser->get("path")->val();
                               easypr::demo::Result result;
                               easypr::demo::accuracyTest(folder.c_str(), result);
                           } else {
                               // single testing
                               auto image = parser->get("path")->val();

                               if (parser->has("color")) {
                                   // return plate color
                                   const char *colors[2] = {"blue", "yellow"};
                                   std::cout
                                           << colors[easypr::api::get_plate_color(image.c_str())]
                                           << std::endl;
                               } else {
                                   // return strings
                                   auto svm = parser->get("svm")->val();
                                   auto ann = parser->get("ann")->val();

                                   auto results = easypr::api::plate_recognize(
                                           image.c_str(), svm.c_str(), ann.c_str());
                                   for (auto s : results) {
                                       std::cout << s << std::endl;
                                   }
                               }
                           }
                       } else {
                           std::cout << "option 'file' cannot be empty." << std::endl;
                       }
                   })
            .others([&]() {
                // no case matched, print all commands.
                std::cout << "There are several sub commands listed below, "
                          << "choose one by typing:\n\n"
                          << "    " << easypr::utils::getFileName(argv[0])
                          << " command [options]\n\n"
                          << "The commands are:\n" << std::endl;
                auto subs = options.get_subroutine_list();
                for (auto sub : subs) {
                    fprintf(stdout, "%s    %s\n", sub.first.c_str(), sub.second.c_str());
                }
                std::cout << std::endl;
            });
}
//only QR cut
void QR_cut(){
    for(int a=1;a<21;a++){
        for(int b=0;b<6;b++){
            fstream _file;
//            string name=to_string(a)+"_"+to_string(b)+"_0.png";
            string name="2_2_0.png";
//            _file.open("resources/robot_test/"+name,ios::in);
            if(!_file)
                continue;
            cv::Mat src = cv::imread("resources/robot_test/"+name);
            Mat greyImg;
            cvtColor(src, greyImg, CV_BGR2GRAY);

//    if (1) {
//        imshow("bin", greyImg);
//        waitKey(0);
//        destroyWindow("bin");
//    }

            Mat binImg;
            binImg = greyImg.clone();

            int width = binImg.cols / 4;
            int height = binImg.rows / 2;

            // iterate through grid
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 4; j++) {
                    Mat src_cell = Mat(binImg, Range(i * height, (i + 1) * height), Range(j * width, (j + 1) * width));
                    cv::threshold(src_cell, src_cell, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);

                }
            }

            rectangle(binImg, Point(0, 0), Point(binImg.cols , binImg.rows ), Scalar(0), 2);

            if (1) {
                imshow("bin", binImg);
                waitKey(0);
                destroyWindow("bin");
            }
//            rectangle(binImg, Point(0, 0), Point(binImg.cols - 1, binImg.rows - 1), Scalar(0), 10,20);

            width = binImg.cols;
            height = binImg.rows;
            int *valArry = new int[width];//创建用于储存每列白色像素个数的数组
            memset(valArry, 0, width * 4);//初始化数组

            int pixelValue;//每个像素的值

            int startIndex = 0;//记录进入字符区的索引
            int preStartIndex = width*0.6;
            int endIndex = 0;//记录进入空白区域的索引
            bool inBlock = false;//是否遍历到了字符区内
            int count = 0;
            Mat roiImg;

            for (int col = 7; col < width; col++)//列
            {
                for (int row = 0; row < height; row++)//行
                {
                    pixelValue = binImg.at<uchar>(row, col);
                    if (pixelValue == 0)//如果是白底黑字
                    {
                        valArry[col]++;
                    }
                }
                float tmp = (float) valArry[col] / (float) binImg.rows;
                if (!inBlock && tmp >= 0.95 && tmp <= 1) {
                    inBlock = true;
                    cout << "startIndex is " << col << endl;
                    startIndex = col ;
                    roiImg = binImg(Range(0, height), Range(startIndex, width));
                    bitwise_not(roiImg, roiImg);
                    cv::copyMakeBorder( roiImg,roiImg, 0, 0, 0, 3, cv::BORDER_CONSTANT, 255);
                    imshow(name,  roiImg);
                    waitKey(0);
                    destroyWindow(name);
                    cv::imwrite("resources/ocr_bin_output/"+name, roiImg);
                    break;
                }
            }
        }
    }
}

void del_edge(Mat input, Mat& result){
    imshow("pre", input);
    waitKey(0);
    Mat greyImg;
    cvtColor(input, greyImg, CV_BGR2GRAY);

    if (0) {
        imshow("grey", greyImg);
        waitKey(0);
//            destroyWindow("grey");
    }

    Mat binImg;
    cv::threshold(greyImg, binImg, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
//    rectangle(binImg, Point(0, 0), Point(binImg.cols , binImg.rows ), Scalar(0), 10);

    if (1) {
        imshow("bin", binImg);
        waitKey(0);
//      destroyWindow("bin");
    }

    int width = input.cols;
    int height = input.rows;
    int *valArry = new int[height];//创建用于储存每列白色像素个数的数组
    memset(valArry, 0, height * 4);//初始化数组

    int pixelValue;//每个像素的值

    int startIndex = 0;//记录进入字符区的索引
    int endIndex = 0;//记录进入空白区域的索引
    int col_tmp = width * 0.25;
    int width_tmp = width * 0.75;
    int aaa = width_tmp-col_tmp;

    //去掉上下边界
    for (int row = 0; row < height; row++)//遍历每个像素点
    {

        for (int col = col_tmp; col < width_tmp; col++) {
            pixelValue = binImg.at<uchar>(row, col);
            if (pixelValue == 0)//如果是白底黑字
            {
                valArry[row]++;
            }
        }
        if(row == height/2){
            for (int row_tmp = row; row_tmp >=0 ; row_tmp--){
                if(valArry[row_tmp]==aaa){
                    if(row_tmp<3)
                        startIndex=0;
                    else
                        startIndex=row_tmp-3;
                    break;
                }
            }
        }
        else if(row > height/2){
            if(valArry[row]==aaa){
                if(row>=height-3)
                    endIndex=height-1;
                else
                    endIndex=row+3;
                break;
            }
        }
    }
//    rectangle(greyImg, Point(0, 0), Point(roiImg.cols , roiImg.rows ), Scalar(0), 10);
    Mat greyImg2 = greyImg(Range(startIndex, endIndex), Range(0, width));


    Mat binImg2;
    cv::threshold(greyImg2, binImg, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
    imshow("horizontal", binImg);

    width = binImg.cols;
    height = binImg.rows;
    valArry = new int[width];//创建用于储存每列白色像素个数的数组
    memset(valArry, 0, width * 4);//初始化数组

    startIndex = 0;//记录进入字符区的索引
//切二维码
    for (int col = 30; col < width; col++)//列
    {
        for (int row = 0; row < height; row++)//行
        {
            pixelValue = binImg.at<uchar>(row, col);
            if (pixelValue == 0)//如果是白底黑字
            {
                valArry[col]++;
            }
        }
        if (valArry[col]==height) {
            startIndex = col ;
            break;
        }
    }
    int flag = 0;
    for (int col = width-1; col > width/2; col--)
    {
        for (int row = 0; row < height; row++)//行
        {
            pixelValue = binImg.at<uchar>(row, col);
            if (pixelValue == 0)//如果是白底黑字
            {
                valArry[col]++;
            }
        }
        if(flag == 0 && valArry[col]==height){
            flag=1;
            continue;
        }
        else if(flag ==1 && valArry[col]<height){
            if(col<width-4)
                endIndex = col + 4;
            else
                endIndex = width - 1;
            Mat greyImg3 = greyImg2(Range(0, height), Range(startIndex, endIndex));
            cv::threshold(greyImg3, binImg, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
            imshow("result", binImg);
            waitKey(0);
            break;
        }
    }
    result = binImg;

}



int crop(Mat input, Mat& output, vector<Point> contours) {
    CvPoint2D32f rectpoint[4];
    CvBox2D rect = minAreaRect(Mat(contours));
    cvBoxPoints(rect, rectpoint); //获取4个顶点坐标
    //与水平线的角度
    float angle = rect.angle;
//    cout << angle << endl;
    int line1 = sqrt((rectpoint[1].y - rectpoint[0].y) * (rectpoint[1].y - rectpoint[0].y) +
                     (rectpoint[1].x - rectpoint[0].x) * (rectpoint[1].x - rectpoint[0].x));
    int line2 = sqrt((rectpoint[3].y - rectpoint[0].y) * (rectpoint[3].y - rectpoint[0].y) +
                     (rectpoint[3].x - rectpoint[0].x) * (rectpoint[3].x - rectpoint[0].x));
    //为了让正方形横着放，所以旋转角度是不一样的。竖放的，给他加90度，翻过来
    if (line1 > line2) {
        angle = 90 + angle;
    }
    Rect mr = boundingRect(Mat(contours));

    Mat RatationedImg(input.rows, input.cols, CV_8UC1);
    RatationedImg.setTo(0);

    //放射变化
    Mat M2 = getRotationMatrix2D(rect.center, angle, 1);//计算旋转加缩放的变换矩阵
    warpAffine(input, RatationedImg, M2, input.size(), 1, 0, Scalar(0));//仿射变换

    if (0) {
        imshow("spin", RatationedImg);
        waitKey(0);
//        destroyWindow("spin");
    }
    // 旋转之后crop
    Mat cutImg(RatationedImg, mr);
    output = cutImg;
    return 1;
}

void crop_robot(){
    cv::Mat src_tmp, result;
    easypr::CCharsSegment plate;
    string s;
    ifstream inf("resources/ocr_demo_pics/label");
    int a,b,c,d,e,f,g,h;
    string name ;
    int count=0;
    while (getline(inf, s))      //getline(inf,s)是逐行读取inf中的文件信息
    {
        if(s[0]=='i'){
            name="";
            for(int i=14;s[i]!='.';i++)
            name+=s[i];
            count=0;
        }
        if(s[0] != 'i'&& s[0] != NULL){
            vector<Point> contours;
            cv::Mat src = cv::imread("resources/ocr_demo_pics/"+name+".jpg");
            sscanf(s.c_str(), "%d,%d %d,%d %d,%d %d,%d", &a,&b,&c,&d, &e, &f,&g,&h);
            contours.push_back(Point2f(a*2,b*2));
            contours.push_back(Point2f(c*2,d*2));
            contours.push_back(Point2f(e*2,f*2));
            contours.push_back(Point2f(g*2,h*2));
//            int result = plate.charsSegment_me(src, resultVec, contours);
            int tmp = crop(src, src_tmp, contours);
            del_edge(src_tmp, result);
            cv::imwrite("resources/crop_robot_output/"+name+"_"+to_string(count)+".png", result);
            count++;
        }
    }
}

int main(int argc, const char* argv[]) {
//    QR_cut();
    crop_robot();


//    std::shared_ptr<easypr::Kv> kv(new easypr::Kv);
//    kv->load("resources/text/chinese_mapping");
//
//    if (argc > 1) {
//        // handle command line execution.
//        command_line_handler(argc, argv);
//        return 0;
//    }
//
//    bool isExit = false;
//    while (!isExit) {
//        easypr::Utils::print_file_lines("resources/text/main_menu");
//        std::cout << kv->get("make_a_choice") << ":";
//
//        int select = 1;
//        bool isRepeat = true;
//        while (isRepeat) {
//            std::cin >> select;
//            isRepeat = false;
//            switch (select) {
//                case 1:
//                    easypr::demo::testMain();
//                    break;
//                case 2:
//                    easypr::demo::accuracyTestMain();
//                    break;
//                case 3:
//                    std::cout << "Run \"demo svm\" for more usage." << std::endl;
//                    {
//                        easypr::SvmTrain svm("tmp/svm", "tmp/svm.xml");
//                        svm.train();
//                    }
//                    break;
//                case 4:
//                    std::cout << "Run \"demo ann\" for more usage." << std::endl;
//                    {
//                        easypr::AnnTrain ann("tmp/ann", "tmp/ann.xml");
//                        ann.train();
//                    }
//                    break;
//                case 5:
//                    easypr::demo::trainChineseMain();
//                    break;
//                case 6: {
//                    //TODO: genenrate gray characters
//                    easypr::demo::accuracyCharRecognizeTest("resources/image/tmp/plates_200k");
//                    break;
//                }
//                case 7: {
//                    easypr::Utils::print_file_lines("resources/text/thanks");
//                    break;
//                }
//                case 8:
//                    isExit = true;
//                    break;
//                default:
//                    std::cout << kv->get("input_error") << ":";
//                    isRepeat = true;
//                    break;
//            }
//        }
//    }

    return 0;
}