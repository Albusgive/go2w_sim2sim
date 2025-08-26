#pragma once

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#define LINUX_GAME // linux下游戏手柄
/*
sudo apt-get install parcellite
sudo apt-get install libudev-dev
sudo apt-get install joystick
target_link_libraries(your_project/your_node udev)
*/

#ifdef LINUX_GAME
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <linux/input.h>
#include <linux/joystick.h>
#include <libudev.h>
#include "gamepadkey.h"

// 解码类型枚举
enum DeviceType
{
    NoDevice,
    XboxOnePad,
    XboxOneWireless,
    NintendoSwitch,
    BEITONG,
    SONY
};

// 手柄按键映射
struct GamePadValues
{
    int time = 0;
    int a = 0;
    int b = 0;
    int x = 0;
    int y = 0;
    int lb = 0;
    int rb = 0;
    int start = 0; // view
    int menu = 0;  //
    int home = 0;
    int screenhot = 0;
    int lo = 0;
    int ro = 0;

    int lx = 0;
    int ly = 0;
    int rx = 0;
    int ry = 0;
    int lt = XboxOnePad_AXIS_VAL_MIN;
    int rt = XboxOnePad_AXIS_VAL_MIN;
    int xx = 0;
    int yy = 0;
};

struct InputDevice
{
    struct udev_device *dev = NULL;
    std::string name = "null";
    std::string type = "null"; // 设备类型usb|2.4G  bluetooth
    std::string path = "null"; // /dev/input/jsx
    DeviceType decode_type = NoDevice;
    std::string parent_sysname; // input和event同一个分支标识
    std::string PID = "null";    //如果是USB设备则有PID和VID
    std::string VID = "null";
};

class GamePad
{
public:
    GamePad();
    ~GamePad();
    /*
    返回map<端口路径，详细信息>
    */
    std::map<std::string, std::string> findGamePad();
    /*
    打印搜索到的手柄
    */
    void showGamePads();
    std::map<std::string, std::string> GamePadpads;
    /*
    无参数默认开启 0号手柄
    有参开启port
    */
    int openGamePad();
    int openGamePad(std::string device_name);
    int openGamePad(int num);
    /*
    绑定触发函数
    */
    void bindGamePadValues(std::function<void(GamePadValues)> callback);
    /*
    解绑触发函数
    */
    void unbindGamePadValues();
    /*
    开始读取GamePad手柄数据
    */
    void readGamePad();
    /*
    停止读取GamePad手柄数据
    */
    void unreadGamePad();

private:
    std::vector<InputDevice> dev_js;
    std::vector<InputDevice> dev_evnet;
    std::function<void(GamePadValues)> lambda;
    bool is_bind = false;
    int GamePadfd;
    GamePadValues xbox_values;
    int port = -1;
    std::thread read_thread;
    std::mutex mtx;
    std::atomic_bool is_running{false};
    DeviceType device_type;
    void Rec();
    bool getdecodeType(InputDevice DecodeName);
    void decodeXboxOnePad(struct js_event js);
    void decodeXboxOneWireless(struct js_event js);
    void decodeNintendoSwitch(struct js_event js);
    void decodeBEITONG(struct js_event js);
    void decodeSony(struct js_event js);
};
#endif
