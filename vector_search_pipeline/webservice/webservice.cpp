
#include "webservice.hpp"

// Session

void Session::run(){
    net::dispatch(stream.get_executor(),beast::bind_front_handler(&Session::read,shared_from_this()));
}

void Session::read() {
    request = {};
    // stream.expires_after(std::chrono::seconds(30)); // timeout
    http::async_read(stream, buffer, request, beast::bind_front_handler(&Session::read_handler,shared_from_this()));
}

void Session::read_handler(beast::error_code ec,std::size_t sz){
    boost::ignore_unused(sz);

    if(ec == http::error::end_of_stream){
        close();
        return;
    }

    if (!ec) {
        write(request_handler(std::move(request)));
    } else {
        std::cerr << "Error while reading: " << ec.message() << std::endl;
        close();
        return;
    }
}

void Session::write(http::response<http::string_body>&& res) {
    auto to_write = std::make_shared<http::response<http::string_body>>(std::move(res));
    http::async_write(stream, *to_write, beast::bind_front_handler(&Session::write_handler,shared_from_this(),to_write));
}

void Session::write_handler(std::shared_ptr<http::response<http::string_body>> to_write, beast::error_code ec,std::size_t sz){
    boost::ignore_unused(sz);

    if(ec){
        std::cerr << "Error while writing: " << ec.message() << std::endl;
        close();
        return;
    }

    read();
}

void Session::close(){
    beast::error_code ec;
    stream.socket().shutdown(btcp::socket::shutdown_send, ec);
}

// Listener

Listener::Listener(net::io_context& ioc, btcp::endpoint endpoint, request_handler_t request_handler) : ioc(ioc), acceptor(net::make_strand(ioc)), request_handler(request_handler) {
    beast::error_code ec;

    // open the acceptor
    acceptor.open(endpoint.protocol(), ec);
    if (ec) {
        std::cerr << "Open error: " << ec.message() << std::endl;
        return;
    }

    // allow address reuse
    acceptor.set_option(net::socket_base::reuse_address(true), ec);
    if (ec) {
        std::cerr << "Set option error: " << ec.message() << std::endl;
        return;
    }

    // bind to the server address
    acceptor.bind(endpoint, ec);
    if (ec) {
        std::cerr << "Bind error: " << ec.message() << std::endl;
        return;
    }

    // start listening for connections
    acceptor.listen(net::socket_base::max_listen_connections, ec);
    if (ec) {
        std::cerr << "Listen error: " << ec.message() << std::endl;
        return;
    }
}

void Listener::run(){
    accept();
}

void Listener::accept() {
    acceptor.async_accept(net::make_strand(ioc),beast::bind_front_handler(&Listener::accept_handler,shared_from_this()));
}

void Listener::accept_handler(beast::error_code ec, btcp::socket socket){
    if(ec){
        std::cerr << "Accept error: " << ec.message() << std::endl;
        return;
    }

    auto session = std::make_shared<Session>(std::move(socket),request_handler);
    session->run();
        
    accept();
}

