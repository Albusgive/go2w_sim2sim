#include "gamepad.h"

GamePad::GamePad()
{
}

GamePad::~GamePad()
{
    unreadGamePad();
    close(GamePadfd);
}

std::map<std::string, std::string> GamePad::findGamePad()
{
    GamePadpads.clear();
    dev_js.clear();
    dev_evnet.clear();
    struct udev *udev_usb;
    struct udev_enumerate *enumerate;
    struct udev_list_entry *devices, *dev_list_entry;
    struct udev_device *dev;

    struct udev *udev_input;
    struct udev_enumerate *enumerate_input;
    struct udev_list_entry *devices_input, *dev_list_entry_input;
    struct udev_device *dev_input;

    // 创建 udev_usb 对象
    udev_usb = udev_new();
    if (!udev_usb)
    {
        fprintf(stderr, "无法创建 udev\n");
        return GamePadpads;
    }
    // 创建 udev_input 对象
    udev_input = udev_new();
    if (!udev_input)
    {
        fprintf(stderr, "无法创建 input udev\n");
        return GamePadpads;
    }

    // 创建一个列举器对象
    enumerate = udev_enumerate_new(udev_usb);
    udev_enumerate_add_match_subsystem(enumerate, "usb");
    udev_enumerate_scan_devices(enumerate);
    devices = udev_enumerate_get_list_entry(enumerate);

    // 创建一个input列举器对象
    enumerate_input = udev_enumerate_new(udev_input);
    udev_enumerate_add_match_subsystem(enumerate_input, "input");
    udev_enumerate_scan_devices(enumerate_input);
    devices_input = udev_enumerate_get_list_entry(enumerate_input);

    // 遍历所有找到的input设备
    udev_list_entry_foreach(dev_list_entry_input, devices_input)
    {
        const char *path;
        // 获取设备的唯一路径
        path = udev_list_entry_get_name(dev_list_entry_input);
        dev_input = udev_device_new_from_syspath(udev_input, path);
        const char *node = udev_device_get_devnode(dev_input);
        if (node != NULL)
        {
            struct udev_device *p = udev_device_get_parent(dev_input);
            // const char *str2 = udev_device_get_subsystem(p);
            // printf("parent sysname:%s\n", node);
            // printf("    parent type:%s\n", udev_device_get_sysname(p));
            if (((std::string)node).find("js") != std::string::npos)
            {
                // 在设备树上向上追溯bluetooth设备获取
                struct udev_device *parent = udev_device_get_parent_with_subsystem_devtype(
                    dev_input, "bluetooth", NULL);
                if (parent)
                {
                    // 保存手柄
                    InputDevice input;
                    input.dev = dev_input;
                    input.path = (std::string)node;
                    input.type = "ble";
                    // 上一级父设备，不然设备名搜不到,因为js和event是同一支个下两个分支
                    const char *str = udev_device_get_sysname(p);
                    if (str != NULL)
                    {
                        input.parent_sysname = (std::string)str;
                    }
                    dev_js.push_back(input);
                }
                else
                {
                    // 在设备树上向上追溯usb设备获取VID VID相同即为同一设备
                    parent = udev_device_get_parent_with_subsystem_devtype(
                        dev_input, "usb", "usb_device");
                    if (parent)
                    {
                        // 保存手柄
                        InputDevice input;
                        input.dev = dev_input;
                        input.path = node;
                        if (input.type != "ble")
                        {
                            input.type = "usb";
                            const char *str = udev_device_get_sysattr_value(parent, "idProduct");
                            if (str != NULL)
                            {
                                input.PID = (std::string)str;
                            }
                            str = udev_device_get_sysattr_value(parent, "idVendor");
                            if (str != NULL)
                            {
                                input.VID = (std::string)udev_device_get_sysattr_value(parent, "idVendor");
                            }
                            // 上一级父设备，不然设备名搜不到,因为js和event是同一支个下两个分支
                            str = udev_device_get_sysname(p);
                            if (str != NULL)
                            {
                                input.parent_sysname = (std::string)str;
                            }
                            dev_js.push_back(input);
                        }
                    }
                }
            }
            else
            {
                // 保存event
                InputDevice input;
                input.dev = dev_input;
                input.path = node;
                if (p != NULL)
                {
                    const char *str = udev_device_get_sysname(p);
                    if (str != NULL)
                    {
                        input.parent_sysname = (std::string)str;
                    }
                    dev_evnet.push_back(input);
                }
            }
        }
        udev_device_unref(dev_input);
    }

    // 寻找js和event相同父设备的USB设备
    for (auto js = dev_js.begin(); js != dev_js.end(); js++)
    {
        for (auto ev : dev_evnet)
        {
            if (js->parent_sysname == ev.parent_sysname)
            {
                // 寻找驱动名称
                int fd = open(ev.path.c_str(), O_RDONLY);
                if (fd < 0)
                {
                    close(fd);
                    continue;
                }
                char name[256] = "null";
                if (ioctl(fd, EVIOCGNAME(sizeof(name) - 1), &name) < 0)
                {
                    // std::cerr << "无法获取设备名称" << std::endl;
                    close(fd);
                }
                else
                {
                    // std::cout << "设备名称: " << name << std::endl;
                    js->name = name;
                    // ble设备直接名字标识
                    if (js->type == "ble")
                    {
                        std::string description(js->name + " | " + "BlUETOOTH DEVICE");
                        GamePadpads.insert(std::pair<std::string, std::string>(js->path, description));
                    }
                    close(fd);
                    break;
                }
            }
        }
    }
    // 遍历所有找到的usb设备,是为了描述信息
    udev_list_entry_foreach(dev_list_entry, devices)
    {
        const char *path;
        // 获取设备的唯一路径
        path = udev_list_entry_get_name(dev_list_entry);
        dev = udev_device_new_from_syspath(udev_usb, path);
        const char *node = udev_device_get_devnode(dev);
        const char *pid = udev_device_get_sysattr_value(dev, "idProduct");
        std::string Pid, Vid;
        if (pid == NULL)
        {
            continue;
        }
        else
        {
            Pid = (std::string)pid;
        }
        const char *vid = udev_device_get_sysattr_value(dev, "idVendor");
        if (vid == NULL)
        {
            continue;
        }
        else
        {
            Vid = (std::string)vid;
        }
        if (node != NULL)
        {
            for (auto it : dev_js)
            {
                if (it.PID == Pid && it.VID == Vid)
                {
                    std::string description(it.name + " | " + "USB DEVICE: " + (std::string)udev_device_get_sysattr_value(dev, "manufacturer") + "| " +
                                            (std::string)udev_device_get_sysattr_value(dev, "product"));
                    GamePadpads.insert(std::pair<std::string, std::string>(it.path, description));
                    break;
                }
            }
        }
        udev_device_unref(dev);
    }
    // 清理
    udev_enumerate_unref(enumerate);
    udev_unref(udev_usb);
    udev_enumerate_unref(enumerate_input);
    udev_unref(udev_input);
    return GamePadpads;
}

void GamePad::showGamePads()
{
    findGamePad();
    if (GamePadpads.empty())
    {
        std::cout << "未找到GamePad" << std::endl;
    }
    for (auto it : GamePadpads)
    {
        int name = it.second.find("|");
        std::cout << "设备名称：" << it.second.substr(0, name) << std::endl;
        std::cout << "  ∟﹣﹣﹣设备路径：" << it.first << std::endl;
        std::cout << "  ∟﹣﹣﹣详细信息：" << it.second.substr(name + 2, it.second.size()) << std::endl;
    }
    std::cout << std::endl;
}

int GamePad::openGamePad()
{
    std::string port("/dev/input/js0");
    for (auto it : dev_js)
    {
        if (it.path.find(port) != std::string::npos)
        {
            GamePadfd = open(port.c_str(), O_RDONLY);
            if (GamePadfd < 0)
            {
                perror("open");
                return -1;
            }
            else
            {
                // 判断解码类型
                getdecodeType(it);
                break;
            }
        }
    }
    return GamePadfd;
}

int GamePad::openGamePad(std::string device_name)
{
    if (device_name.size() < 3)
    {
        // 写入数字
        std::string port("/dev/input/js");
        port += device_name;
        GamePadfd = open(port.c_str(), O_RDONLY);
        if (GamePadfd < 0)
        {
            perror("open");
            return -1;
        }
        else
        {
            // 判断解码类型
            for (auto it : dev_js)
            {
                if (it.path == port)
                {
                    bool is = getdecodeType(it);
                    if (!is)
                    {
                        std::cout << "no find decode type" << std::endl;
                        return -1;
                    }
                    break;
                }
            }
            return GamePadfd;
        }
    }
    for (auto it : dev_js)
    {
        if (it.name.find(device_name) != std::string::npos)
        {
            GamePadfd = open(it.path.c_str(), O_RDONLY);
            if (GamePadfd < 0)
            {
                break;
            }
            else
            {
                // 判断解码类型
                getdecodeType(it);
                break;
            }
        }
    }
    for (auto it : dev_js)
    {
        if (it.path.find(device_name) != std::string::npos)
        {
            GamePadfd = open(it.path.c_str(), O_RDONLY);
            if (GamePadfd < 0)
            {
                perror("open");
                return -1;
            }
            else
            {
                // 判断解码类型
                getdecodeType(it);
                break;
            }
        }
    }
    return GamePadfd;
}

int GamePad::openGamePad(int num)
{
    port = num;
    std::string port("/dev/input/js");
    port += std::to_string(num);
    for (auto it : dev_js)
    {
        if (it.path.find(port) != std::string::npos)
        {
            GamePadfd = open(port.c_str(), O_RDONLY);
            if (GamePadfd < 0)
            {
                perror("open");
                return -1;
            }
            else
            {
                // 判断解码类型
                getdecodeType(it);
                break;
            }
        }
    }
    return GamePadfd;
}

void GamePad::bindGamePadValues(std::function<void(GamePadValues)> callback)
{
    lambda = callback;
    is_bind = true;
}

void GamePad::unbindGamePadValues()
{
    unreadGamePad();
    is_bind = false;
}

void GamePad::readGamePad()
{
    if (!is_bind)
    {
        std::cout << "no bind callback" << std::endl;
        return;
    }
    if (is_running.load())
    {
        std::cout << "GamePad is running" << std::endl;
        return;
    }
    if (port >= 0)
    {
        openGamePad(port);
    }
    is_running.store(true);
    read_thread = std::thread(std::bind(&GamePad::Rec, this));
    read_thread.detach();
}

void GamePad::unreadGamePad()
{
    is_running.store(false);
    if (close(GamePadfd) < 0)
    {
        // perror("close");
    }
}

void GamePad::Rec()
{
    while (is_running.load())
    {
        std::unique_lock<std::mutex> lock(mtx);
        int len;
        struct js_event js;
        len = read(GamePadfd, &js, sizeof(struct js_event));
        lock.unlock();
        if (len < 0)
        {
            perror("read");
            continue;
        }
        switch (device_type)
        {
        case XboxOnePad:
            decodeXboxOnePad(js);
            break;
        case XboxOneWireless:
            decodeXboxOneWireless(js);
            break;
        case NintendoSwitch:
            decodeNintendoSwitch(js);
            break;
        case BEITONG:
            decodeBEITONG(js);
            break;
        case SONY:
            decodeSony(js);
            break;
        default:
            break;
        }
        lock.lock();
        lambda(xbox_values);
        lock.unlock();
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    lambda(GamePadValues());
}

bool GamePad::getdecodeType(InputDevice DecodeName)
{
    std::cout << "decode type: ";
    if (DecodeName.name.find("X-Box One pad") != std::string::npos)
    {
        std::cout << "XboxOnePad" << std::endl;
        device_type = XboxOnePad;
    }
    else if (DecodeName.name.find("Xbox Wireless") != std::string::npos)
    {
        std::cout << "XboxOneWireless" << std::endl;
        device_type = XboxOneWireless;
    }
    else if (DecodeName.name.find("Nintendo Switch Pro") != std::string::npos)
    {
        std::cout << "NintendoSwitch" << std::endl;
        device_type = NintendoSwitch;
    }
    else if (DecodeName.name.find("BEITONG") != std::string::npos)
    {
        std::cout << "BEITONG" << std::endl;
        device_type = BEITONG;
    }
    else if (DecodeName.name.find("Sony") != std::string::npos)
    {
        std::cout << "Sony" << std::endl;
        device_type = SONY;
    }
    else if (DecodeName.type == "usb")
    {
        std::cout << "USB" << std::endl;
        device_type = XboxOnePad;
    }
    else if (DecodeName.type == "ble")
    {
        std::cout << "ble" << std::endl;
        device_type = XboxOneWireless;
    }
    else
    {
        return false;
    }
    return true;
}

void GamePad::decodeXboxOnePad(js_event js)
{
    int type, number, value;
    type = js.type;
    number = js.number;
    value = js.value;
    xbox_values.time = js.time;
    if (type == JS_EVENT_BUTTON)
    {
        switch (number)
        {
        case XboxOnePad_BUTTON_A:
            xbox_values.a = value;
            break;

        case XboxOnePad_BUTTON_B:
            xbox_values.b = value;
            break;

        case XboxOnePad_BUTTON_X:
            xbox_values.x = value;
            break;

        case XboxOnePad_BUTTON_Y:
            xbox_values.y = value;
            break;

        case XboxOnePad_BUTTON_LB:
            xbox_values.lb = value;
            break;

        case XboxOnePad_BUTTON_RB:
            xbox_values.rb = value;
            break;

        case XboxOnePad_BUTTON_START:
            xbox_values.start = value;
            break;

        case XboxOnePad_BUTTON_MENU:
            xbox_values.menu = value;
            break;

        case XboxOnePad_BUTTON_HOME:
            xbox_values.home = value;
            break;

        case XboxOnePad_BUTTON_LO:
            xbox_values.lo = value;
            break;

        case XboxOnePad_BUTTON_RO:
            xbox_values.ro = value;
            break;

        default:
            break;
        }
    }
    else if (type == JS_EVENT_AXIS)
    {
        switch (number)
        {
        case XboxOnePad_AXIS_LX:
            xbox_values.lx = value;
            break;

        case XboxOnePad_AXIS_LY:
            xbox_values.ly = value;
            break;

        case XboxOnePad_AXIS_RX:
            xbox_values.rx = value;
            break;

        case XboxOnePad_AXIS_RY:
            xbox_values.ry = value;
            break;

        case XboxOnePad_AXIS_LT:
            xbox_values.lt = value;
            break;

        case XboxOnePad_AXIS_RT:
            xbox_values.rt = value;
            break;

        case XboxOnePad_AXIS_XX:
            xbox_values.xx = value;
            break;

        case XboxOnePad_AXIS_YY:
            xbox_values.yy = value;
            break;

        default:
            break;
        }
    }
}

void GamePad::decodeXboxOneWireless(js_event js)
{
    int type, number, value;
    type = js.type;
    number = js.number;
    value = js.value;
    xbox_values.time = js.time;
    if (type == JS_EVENT_BUTTON)
    {
        switch (number)
        {
        case XboxOneWireless_BUTTON_A:
            xbox_values.a = value;
            break;

        case XboxOneWireless_BUTTON_B:
            xbox_values.b = value;
            break;

        case XboxOneWireless_BUTTON_X:
            xbox_values.x = value;
            break;

        case XboxOneWireless_BUTTON_Y:
            xbox_values.y = value;
            break;

        case XboxOneWireless_BUTTON_LB:
            xbox_values.lb = value;
            break;

        case XboxOneWireless_BUTTON_RB:
            xbox_values.rb = value;
            break;

        case XboxOneWireless_BUTTON_START:
            xbox_values.start = value;
            break;

        case XboxOneWireless_BUTTON_MENU:
            xbox_values.menu = value;
            break;

        case XboxOneWireless_BUTTON_HOME:
            xbox_values.home = value;
            break;

        case XboxOneWireless_BUTTON_LO:
            xbox_values.lo = value;
            break;

        case XboxOneWireless_BUTTON_RO:
            xbox_values.ro = value;
            break;

        default:
            break;
        }
    }
    else if (type == JS_EVENT_AXIS)
    {
        switch (number)
        {
        case XboxOneWireless_AXIS_LX:
            xbox_values.lx = value;
            break;

        case XboxOneWireless_AXIS_LY:
            xbox_values.ly = value;
            break;

        case XboxOneWireless_AXIS_RX:
            xbox_values.rx = value;
            break;

        case XboxOneWireless_AXIS_RY:
            xbox_values.ry = value;
            break;

        case XboxOneWireless_AXIS_LT:
            xbox_values.lt = value;
            break;

        case XboxOneWireless_AXIS_RT:
            xbox_values.rt = value;
            break;

        case XboxOneWireless_AXIS_XX:
            xbox_values.xx = value;
            break;

        case XboxOneWireless_AXIS_YY:
            xbox_values.yy = value;
            break;

        default:
            break;
        }
    }
}

void GamePad::decodeNintendoSwitch(js_event js)
{
    int type, number, value;
    type = js.type;
    number = js.number;
    value = js.value;
    xbox_values.time = js.time;
    if (type == JS_EVENT_BUTTON)
    {
        switch (number)
        {
        case NintendoSwitchPro_BUTTON_A:
            xbox_values.a = value;
            break;

        case NintendoSwitchPro_BUTTON_B:
            xbox_values.b = value;
            break;

        case NintendoSwitchPro_BUTTON_X:
            xbox_values.x = value;
            break;

        case NintendoSwitchPro_BUTTON_Y:
            xbox_values.y = value;
            break;

        case NintendoSwitchPro_BUTTON_LB:
            xbox_values.lb = value;
            break;

        case NintendoSwitchPro_BUTTON_RB:
            xbox_values.rb = value;
            break;

        case NintendoSwitchPro_BUTTON_VIEW:
            xbox_values.start = value;
            break;

        case NintendoSwitchPro_BUTTON_MENU:
            xbox_values.menu = value;
            break;

        case NintendoSwitchPro_BUTTON_HOME:
            xbox_values.home = value;
            break;

        case NintendoSwitchPro_BUTTON_LO:
            xbox_values.lo = value;
            break;

        case NintendoSwitchPro_BUTTON_RO:
            xbox_values.ro = value;
            break;

        case NintendoSwitchPro_BUTTON_LT:
        {
            if (value)
            {
                xbox_values.lt = NintendoSwitchPro_BUTTON_VAL_DOWN;
            }
            else
            {
                xbox_values.lt = NintendoSwitchPro_BUTTON_VAL_UP;
            }
        }
        break;

        case NintendoSwitchPro_BUTTON_RT:
        {
            if (value)
            {
                xbox_values.rt = NintendoSwitchPro_BUTTON_VAL_DOWN;
            }
            else
            {
                xbox_values.rt = NintendoSwitchPro_BUTTON_VAL_UP;
            }
        }
        break;

        case NintendoSwitchPro_BUTTON_SCREENSHORT:
            xbox_values.screenhot = value;
            break;

        default:
            break;
        }
    }
    else if (type == JS_EVENT_AXIS)
    {
        switch (number)
        {
        case NintendoSwitchPro_AXIS_LX:
            xbox_values.lx = value;
            break;

        case NintendoSwitchPro_AXIS_LY:
            xbox_values.ly = value;
            break;

        case NintendoSwitchPro_AXIS_RX:
            xbox_values.rx = value;
            break;

        case NintendoSwitchPro_AXIS_RY:
            xbox_values.ry = value;
            break;

        case NintendoSwitchPro_AXIS_XX:
            xbox_values.xx = value;
            break;

        case NintendoSwitchPro_AXIS_YY:
            xbox_values.yy = value;
            break;

        default:
            break;
        }
    }
}

void GamePad::decodeBEITONG(js_event js)
{
    int type, number, value;
    type = js.type;
    number = js.number;
    value = js.value;
    xbox_values.time = js.time;
    if (type == JS_EVENT_BUTTON)
    {
        switch (number)
        {
        case BEITONG_BUTTON_A:
            xbox_values.a = value;
            break;

        case BEITONG_BUTTON_B:
            xbox_values.b = value;
            break;

        case BEITONG_BUTTON_X:
            xbox_values.x = value;
            break;

        case BEITONG_BUTTON_Y:
            xbox_values.y = value;
            break;

        case BEITONG_BUTTON_LB:
            xbox_values.lb = value;
            break;

        case BEITONG_BUTTON_RB:
            xbox_values.rb = value;
            break;

        case BEITONG_BUTTON_MENU:
            xbox_values.menu = value;
            break;

        case BEITONG_BUTTON_HOME:
            xbox_values.home = value;
            break;

        case BEITONG_BUTTON_LO:
            xbox_values.lo = value;
            break;

        case BEITONG_BUTTON_RO:
            xbox_values.ro = value;
            break;

        default:
            break;
        }
    }
    else if (type == JS_EVENT_AXIS)
    {
        switch (number)
        {
        case BEITONG_AXIS_LX:
            xbox_values.lx = value;
            break;

        case BEITONG_AXIS_LY:
            xbox_values.ly = value;
            break;

        case BEITONG_AXIS_RX:
            xbox_values.rx = value;
            break;

        case BEITONG_AXIS_RY:
            xbox_values.ry = value;
            break;

        case BEITONG_AXIS_LT:
            xbox_values.lt = value;
            break;

        case BEITONG_AXIS_RT:
            xbox_values.rt = value;
            break;

        case BEITONG_AXIS_XX:
            xbox_values.xx = value;
            break;

        case BEITONG_AXIS_YY:
            xbox_values.yy = value;
            break;

        default:
            break;
        }
    }
}

void GamePad::decodeSony(js_event js)
{
    int type, number, value;
    type = js.type;
    number = js.number;
    value = js.value;
    xbox_values.time = js.time;
    if (type == JS_EVENT_BUTTON)
    {
        switch (number)
        {
        case SONY_BUTTON_A:
            xbox_values.a = value;
            break;

        case SONY_BUTTON_B:
            xbox_values.b = value;
            break;

        case SONY_BUTTON_X:
            xbox_values.x = value;
            break;

        case SONY_BUTTON_Y:
            xbox_values.y = value;
            break;

        case SONY_BUTTON_LB:
            xbox_values.lb = value;
            break;

        case SONY_BUTTON_RB:
            xbox_values.rb = value;
            break;

        case SONY_BUTTON_START:
            xbox_values.start = value;
            break;

        case SONY_BUTTON_MENU:
            xbox_values.menu = value;
            break;

        case SONY_BUTTON_HOME:
            xbox_values.home = value;
            break;

        case SONY_BUTTON_LO:
            xbox_values.lo = value;
            break;

        case SONY_BUTTON_RO:
            xbox_values.ro = value;
            break;
        case SONY_BUTTON_SCREENSHORT:
            xbox_values.screenhot = value;
            break;

        default:
            break;
        }
    }
    else if (type == JS_EVENT_AXIS)
    {
        switch (number)
        {
        case SONY_AXIS_LX:
            xbox_values.lx = value;
            break;

        case SONY_AXIS_LY:
            xbox_values.ly = value;
            break;

        case SONY_AXIS_RX:
            xbox_values.rx = value;
            break;

        case SONY_AXIS_RY:
            xbox_values.ry = value;
            break;

        case SONY_AXIS_LT:
            xbox_values.lt = value;
            break;

        case SONY_AXIS_RT:
            xbox_values.rt = value;
            break;

        case SONY_AXIS_XX:
            xbox_values.xx = value;
            break;

        case SONY_AXIS_YY:
            xbox_values.yy = value;
            break;

        default:
            break;
        }
    }
}
