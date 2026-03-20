#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <boost/algorithm/string/join.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <iostream>
#include <functional>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using btcp = net::ip::tcp;

using request_handler_t = std::function<http::response<http::string_body>(http::request<http::string_body>&&)>;

class Session : public std::enable_shared_from_this<Session> {
    beast::tcp_stream stream;
    beast::flat_buffer buffer;
    http::request<http::string_body> request;
    request_handler_t request_handler;

public:
    Session(btcp::socket&& socket,request_handler_t request_handler) : stream(std::move(socket)), request_handler(request_handler) {}
    void run();

private:
    void read();
    void read_handler(beast::error_code ec,std::size_t sz);

    void write(http::response<http::string_body>&& res);
    void write_handler(std::shared_ptr<http::response<http::string_body>> to_write, beast::error_code ec,std::size_t sz);

    void close();
};

class Listener : public std::enable_shared_from_this<Listener> {
    net::io_context& ioc;
    btcp::acceptor acceptor;
    request_handler_t request_handler;

public:
    Listener(net::io_context& ioc, btcp::endpoint endpoint,request_handler_t request_handler);
    void run();

private:
    void accept();
    void accept_handler(beast::error_code ec, btcp::socket socket);
};

