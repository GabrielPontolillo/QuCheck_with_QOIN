namespace QSharpCheck {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;

    operation Teleport(q0 : Qubit, q1 : Qubit, q2 : Qubit) : Unit {
            // decompose SX(1) into RX(pi/2, 1) and R(PauliI, )
            Rx(PI()/2.0, q1);
            R(PauliI, -PI() / 2.0, q1);

            CNOT(q1, q2);
            CNOT(q0, q1);
            H(q0);
            CNOT(q1, q2);            
            Controlled Z([q0], (q2));
    }

}
