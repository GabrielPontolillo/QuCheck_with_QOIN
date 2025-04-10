namespace QSharpCheck {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;

    operation Teleport(q0 : Qubit, q1 : Qubit, q2 : Qubit) : Unit {
            H(q1);

            // decompose CSX(1, 2) into RX(pi/2, 1) and R(PauliI, )
            Controlled Rx([q1], (PI()/2.0, q2));
            Controlled R([q1], (PauliI, -PI()/2.0, q2));

            CNOT(q1, q2);
            CNOT(q0, q1);
            H(q0);
            CNOT(q1, q2);
            Controlled Z([q0], (q2));
    }
}
