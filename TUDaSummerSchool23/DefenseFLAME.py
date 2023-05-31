from TUDaSummerSchool23.NetworkUtils import *
import socket
from TUDaSummerSchool23.Utils import print_timed

HOST, PORT = "130.83.76.1", 4242

class FLAME:

    def __init__(self, server_host=HOST, server_port=PORT):
        self.description = 'FLAME'
        self.host = server_host
        self.port = server_port

    def __call__(self, global_model_state_dict, all_models, number_of_benign_clients, number_of_malicious_clients):

        print_timed('Serialize Models')
        serializable_model = tensor_to_float(global_model_state_dict)

        serializable_all = []
        for model in all_models:
            serializable_all.append(tensor_to_float(model))

        message = OrderedDict()
        message["global_model"] = serializable_model
        message["all_models"] = serializable_all
        message["number_of_benign_clients"] = number_of_benign_clients
        message["number_of_malicious_clients"] = number_of_malicious_clients

        #print_timed(f'Connect to Server {self.host}')
        # Create a socket (SOCK_STREAM means a TCP socket)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.host, self.port))
            print_timed('Send Request')
            send_msg(sock, message)
            print_timed('Receive Answer')
            received = recv_msg(sock)
            print_timed('Deserialize Answer')
            return float_to_tensor(received)
        except:
            print_timed('An Error occurred')
        finally:
            sock.close()

